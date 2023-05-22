import numpy as np
from lightfm import LightFM
from lightfm._lightfm_fast import CSRMatrix, FastLightFM, fit_bpr, fit_warp
from scipy import sparse
from scipy.stats import rankdata

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

CYTHON_DTYPE = np.float32


class LightFMMPIWrap:
    def __init__(self, model: LightFM) -> None:
        if MPI is None:
            self.model = model
        else:
            self.mpi_world = MPI.COMM_WORLD
            self.mpi_world_size = self.mpi_world.size
            self.mpi_rank = self.mpi_world.Get_rank()
            if self.mpi_rank == 0:
                self.model = model
            else:
                self.model = None

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
        if MPI is None:
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

    def _get_sparse_interactions_to_scatter(
        self, interactions: sparse._csr.csr_matrix, sample_weights: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """ Temporal function as partitioning will be implemented in Spark.
        Extracts data from scipy CSR matrix for further MPI.Scatter operation
        """

        if isinstance(interactions, sparse._csr.csr_matrix):
            interactions = interactions.tocoo()
        if not isinstance(interactions, sparse._coo.coo_matrix):
            raise TypeError(
                "Interactions matrix should be in scipy COO or CSR matrix format"
            )
        row_ids = interactions.row
        col_ids = interactions.col
        matrix_data = interactions.data

        # get ids of MPI workers
        ranks = rankdata(row_ids, method="min")
        split_ids = np.unique(
            ranks // ((np.max(ranks) // self.mpi_world_size) + 1), return_index=True,
        )[1][1:]

        # split data across mpi workers
        user_rows_ids = np.split(row_ids, split_ids)
        item_cols_ids = np.split(col_ids, split_ids)
        interactions_data = np.split(matrix_data, split_ids)
        sample_weights_data = np.split(sample_weights, split_ids)

        return user_rows_ids, item_cols_ids, interactions_data, sample_weights_data

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
        if MPI is None:
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
            rank = self.mpi_world.Get_rank()
            if rank == 0:
                interactions = interactions.tocoo()
                if interactions.dtype != CYTHON_DTYPE:
                    interactions.data = interactions.data.astype(CYTHON_DTYPE)

                sample_weight_data = self.model._process_sample_weight(
                    interactions, sample_weight
                )
                n_users, n_items = interactions.shape
                (user_features, item_features) = self.model._construct_feature_matrices(
                    n_users, n_items, user_features, item_features
                )
                for input_data in (
                    user_features.data,
                    item_features.data,
                    interactions.data,
                    sample_weight_data,
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

                # Convert interactions matrix to list of np.arrays with length of number of MPI nodes
                (
                    rows_sc_interaction,
                    cols_sc_interaction,
                    inter_sc_interaction,
                    sc_sample_weights,
                ) = self._get_sparse_interactions_to_scatter(
                    interactions, sample_weight_data
                )

            else:
                (
                    rows_sc_interaction,
                    cols_sc_interaction,
                    inter_sc_interaction,
                    sc_sample_weights,
                ) = (None, None, None, None)
                user_features, item_features = None, None
                n_users, n_items = None, None

            # temporal (Spark realization later)
            rows_sc_interaction = self.mpi_world.scatter(rows_sc_interaction, root=0)
            cols_sc_interaction = self.mpi_world.scatter(cols_sc_interaction, root=0)
            inter_sc_interaction = self.mpi_world.scatter(inter_sc_interaction, root=0)
            sc_sample_weights = self.mpi_world.scatter(sc_sample_weights, root=0)

            # Broadcast input to MPI workers
            user_features = self.mpi_world.bcast(user_features, root=0)
            item_features = self.mpi_world.bcast(item_features, root=0)
            n_users, n_items = self.mpi_world.bcast((n_users, n_items), root=0)

            # Get local interactions partition in COO sparse matrix format
            mpi_interactions = sparse.coo_matrix(
                (inter_sc_interaction, (rows_sc_interaction, cols_sc_interaction)),
                shape=(n_users, n_items),
            )

            # Initialize model on MPI workers
            self.model = self.mpi_world.bcast(self.model, root=0)
            self._initialize_local_state()

            # Each MPI worker runs on interaction matrix partition
            for _ in self.model._progress(epochs, verbose=verbose):
                self._run_epoch_(
                    item_features,
                    user_features,
                    mpi_interactions,
                    sc_sample_weights,
                    num_threads,
                    self.model.loss,
                )
                self.model._check_finite()

            # TODO delete models from workers?

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
            self.__setattr__(
                attr_name,
                self.mpi_world.allreduce(self.__getattribute__(attr_name), op=MPI.SUM),
            )

        for attr_name in average_attributes:
            attr_value = self.__getattribute__(attr_name)
            self.__setattr__(attr_name, attr_value / self.mpi_world_size)

    def _run_epoch_(
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

        # Get embeddings deltas for the MPI reduction
        self._get_update_delta_after_fit()

        # Perform AllReduce reduction on local states
        self._reduce_states_on_workers()
        # Update local models with common model states
        self._update_model_with_reduced_data()

