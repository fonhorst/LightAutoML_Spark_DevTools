import numpy as np
from lightfm import LightFM
from lightfm._lightfm_fast import CSRMatrix, FastLightFM, fit_bpr, fit_warp
from scipy import sparse
import pyspark
import pygloo.dist.pygloo as pgl
import itertools
import typing as t
import torch.distributed as dist
from datetime import timedelta

CYTHON_DTYPE = np.float32

class LightFMGlooWrap:
    def __init__(
            self,
            model: LightFM,
            use_spark: bool = True,
            world_size: t.Optional[int] = None,
    ) -> None:
        self.model = model
        self.world_size = world_size
        self.use_spark = use_spark
        if self.use_spark and not self.world_size:
            raise ValueError('For Spark usage, world_size (number of Spark executors) must be defined.')

    def fit(
            self,
            interactions,
            user_features=None,
            item_features=None,
            sample_weight=None,
            epochs=1,
            num_threads=1,
            verbose=False,
    ):
        if not self.use_spark:
            self.model.fit_partial(
                interactions,
                user_features,
                item_features,
                sample_weight,
                epochs,
                num_threads,
                verbose,
            )
        else:
            self.model._reset_state()

            return self.fit_partial(
                interactions,
                user_features=user_features,
                item_features=item_features,
                sample_weight=sample_weight,
                epochs=epochs,
                num_threads=num_threads,
                verbose=verbose,
            )

    def _initialize_local_state(self) -> None:
        """ Create local copy of the model states. """
        self.local_item_feature_gradients = self.model.item_embedding_gradients.copy()
        self.local_item_feature_momentum = self.model.item_embedding_momentum.copy()
        self.local_item_bias_gradients = self.model.item_bias_gradients.copy()
        self.local_item_bias_momentum = self.model.item_bias_momentum.copy()
        self.local_user_feature_gradients = self.model.user_embedding_gradients.copy()
        self.local_user_feature_momentum = self.model.user_embedding_momentum.copy()
        self.local_user_bias_gradients = self.model.user_bias_gradients.copy()
        self.local_user_bias_momentum = self.model.user_bias_momentum.copy()

    def rdd_to_csr(self, partition_interactions: itertools.chain, num_users: int, num_items: int):
        user_ids, item_ids, relevance = [], [], []
        for row in partition_interactions:
            user_ids.append(row.user_idx)
            item_ids.append(row.item_idx)
            relevance.append(row.relevance)

        csr = sparse.csr_matrix(
            (relevance, (user_ids, item_ids)),
            shape=(num_users, num_items),
        )
        return csr

    def get_num_users_and_items(self, interactions: pyspark.sql.dataframe.DataFrame):
        num_users = interactions.agg({"user_idx": "max"}).collect()[0][0]
        num_items = interactions.agg({"item_idx": "max"}).collect()[0][0]
        return num_users + 1, num_items + 1

    def fit_partial(
            self,
            interactions,
            user_features=None,
            item_features=None,
            sample_weight=None,
            epochs=1,
            num_threads=1,
            verbose=False,
    ):
        if not self.use_spark:
            print('Warning! No spark executors launched') # TODO logging
            self.model.fit_partial(
                interactions,
                user_features,
                item_features,
                sample_weight,
                epochs,
                num_threads,
                verbose,
            )
        else:
            print('Spark executors launched')  # TODO
            n_users, n_items = self.get_num_users_and_items(interactions)
            # print(n_users, n_items)
            (user_features, item_features) = self.model._construct_feature_matrices(
                n_users, n_items, user_features, item_features
            )

            for input_data in (
                    user_features.data,
                    item_features.data,
            ):
                self.model._check_input_finite(input_data)

            if self.model.item_embeddings is None:
                self.model._initialize(
                    self.model.no_components,
                    item_features.shape[1],
                    user_features.shape[1],
                )

            if not item_features.shape[1] == self.model.item_embeddings.shape[0]:
                raise ValueError("Incorrect number of features in item_features")
            if not user_features.shape[1] == self.model.user_embeddings.shape[0]:
                raise ValueError("Incorrect number of features in user_features")
            if num_threads < 1:
                raise ValueError("Number of threads must be 1 or larger.")

            def udf_to_map_on_interactions_with_index(p_idx, partition_interactions):
                context = pgl.rendezvous.Context(p_idx, self.world_size)
                attr = pgl.transport.tcp.attr("localhost") # TODO
                dev = pgl.transport.tcp.CreateDevice(attr)
                real_store = dist.TCPStore(
                        "localhost",
                        1234,
                        self.world_size,
                        True if p_idx == 0 else False,
                        timedelta(seconds=30)
                    )
                store = pgl.rendezvous.CustomStore(real_store)
                context.connectFullMesh(store, dev)
                self.gloo_context = context

                interactions = self.rdd_to_csr(partition_interactions, num_users=n_users, num_items=n_items)
                interactions = interactions.tocoo()
                if interactions.dtype != CYTHON_DTYPE:
                    interactions.data = interactions.data.astype(CYTHON_DTYPE)

                if not sample_weight:
                    sample_weight_data = self.model._process_sample_weight(
                        interactions, sample_weight
                    )
                else:
                    # TODO: question
                    #  sample weights processing and partitioning
                    #  (how sample weights if they exist are represented in log?)
                    raise NotImplementedError

                for input_data in (
                        interactions.data,
                        sample_weight_data,
                ):
                    self.model._check_input_finite(input_data)

                # Get local interactions partition in COO sparse matrix format
                interactions_part = sparse.coo_matrix(
                    (interactions.data, (interactions.row, interactions.col)),
                    shape=(n_users, n_items),
                )

                # Copy model states to executors
                self._initialize_local_state()

                # Each Spark executor runs on interaction matrix partition
                for _ in self.model._progress(epochs, verbose=verbose):
                    self._run_epoch_spark(
                        item_features,
                        user_features,
                        interactions_part,
                        sample_weight_data,
                        num_threads,
                        self.model.loss,
                    )
                    self.model._check_finite()

                if p_idx == 0:
                    self.gloo_context = None  # TODO disconnect context (+ release resources)
                    yield self

            self = interactions.rdd.mapPartitionsWithIndex(udf_to_map_on_interactions_with_index).collect()[0]
            # TODO: question
            #  is it ok to assign to self?

        return self

    def _copy_represenations_for_update(self) -> None:
        """ Create local copy of the item and user representations. """

        self.local_item_features = self.model.item_embeddings.copy()
        self.local_item_biases = self.model.item_biases.copy()
        self.local_user_features = self.model.user_embeddings.copy()
        self.local_user_biases = self.model.user_biases.copy()

    def _get_lightfm_data(self) -> FastLightFM:
        """ Create FastLightFM class from the states to run update. """

        lightfm_data = FastLightFM(
            self.local_item_features,
            self.local_item_feature_gradients,
            self.local_item_feature_momentum,
            self.local_item_biases,
            self.local_item_bias_gradients,
            self.local_item_bias_momentum,
            self.local_user_features,
            self.local_user_feature_gradients,
            self.local_user_feature_momentum,
            self.local_user_biases,
            self.local_user_bias_gradients,
            self.local_user_bias_momentum,
            self.model.no_components,
            int(self.model.learning_schedule == "adadelta"),
            self.model.learning_rate,
            self.model.rho,
            self.model.epsilon,
            self.model.max_sampled,
        )

        return lightfm_data

    def _get_update_delta_after_fit(self):
        """ Extract initial representation values to get delta from update. """

        self.local_item_features -= self.model.item_embeddings
        self.local_item_biases -= self.model.item_biases
        self.local_user_features -= self.model.user_embeddings
        self.local_user_biases -= self.model.user_biases

    def _update_model_with_reduced_data(self):
        """ Updates model state after MPI operations. """

        self.model.item_embeddings += self.local_item_features
        self.model.item_embedding_gradients = self.local_item_feature_gradients
        self.model.item_embedding_momentum = self.local_item_feature_momentum
        self.model.item_biases += self.local_item_biases
        self.model.item_bias_gradients = self.local_item_bias_gradients
        self.model.item_bias_momentum = self.local_item_bias_momentum
        self.model.user_embeddings += self.local_user_features
        self.model.user_embedding_gradients = self.local_user_feature_gradients
        self.model.user_embedding_momentum = self.local_user_feature_momentum
        self.model.user_biases += self.local_user_biases
        self.model.user_bias_gradients = self.local_user_bias_gradients
        self.model.user_bias_momentum = self.local_user_bias_momentum

    def _reduce_states_on_workers(self):
        """ Perform AllReduce operation summing up representations and averaging the optimization parameters. """

        sum_attributes = (
            "local_user_features",
            "local_user_biases",
        )

        average_attributes = (
            "local_item_features",
            "local_item_biases",
            "local_item_feature_gradients",
            "local_item_feature_momentum",
            "local_item_bias_gradients",
            "local_item_bias_momentum",
            "local_user_feature_gradients",
            "local_user_feature_momentum",
            "local_user_bias_gradients",
            "local_user_bias_momentum",
        )

        for attr_name in sum_attributes + average_attributes:
            sendbuf = self.__getattribute__(attr_name)
            recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
            sendptr = sendbuf.ctypes.data
            recvptr = recvbuf.ctypes.data
            data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
            datatype = pgl.glooDataType_t.glooFloat32

            pgl.allreduce(
                self.gloo_context,
                sendptr,
                recvptr,
                data_size,
                datatype,
                pgl.ReduceOp.SUM,
                pgl.allreduceAlgorithm.RING
            )
            self.__setattr__(attr_name, recvbuf)

        for attr_name in average_attributes:
            attr_value = self.__getattribute__(attr_name)
            self.__setattr__(attr_name, attr_value / self.world_size)

    def _run_epoch_spark(
            self,
            item_features,
            user_features,
            interactions,
            sample_weight,
            num_threads,
            loss,
    ):
        if loss in ("warp", "bpr", "warp-kos"):
            positives_lookup = CSRMatrix(
                self.model._get_positives_lookup_matrix(interactions)
            )
        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        self.model.random_state.shuffle(shuffle_indices)

        # Get representations copies from the local model
        self._copy_represenations_for_update()
        lightfm_data = self._get_lightfm_data()

        if loss == "warp":
            # Run updates on the model state copy
            fit_warp(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.model.learning_rate,
                self.model.item_alpha,  # TODO regulatization
                self.model.user_alpha,
                num_threads,
                self.model.random_state,
            )

        elif loss == "bpr":
            fit_bpr(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.model.learning_rate,
                self.model.item_alpha,  # TODO regulatization
                self.model.user_alpha,
                num_threads,
                self.model.random_state,
            )
        else:
            raise NotImplementedError(
                "Only `warp`, `bpr` losses are available by the moment"
            )

        # Get embeddings deltas before reduction
        self._get_update_delta_after_fit()
        # Perform AllReduce reduction on local states
        self._reduce_states_on_workers()
        # Update local models with common model states
        self._update_model_with_reduced_data()
