from app.flet_app.views.training import (
    TrainingView,
    _DEFAULT_DATA_ROOT,
    _DEFAULT_DYNAMIC_FEATURE_MODE,
    _DEFAULT_DYNAMIC_MODEL_TYPE,
    _DEFAULT_DYNAMIC_MODEL_OUT,
    _DEFAULT_DYNAMIC_RECORD_FRAMES,
    _DEFAULT_DYNAMIC_RECORD_SAMPLES,
    _DEFAULT_NEGATIVE_SAMPLES_PER_LABEL,
    _DEFAULT_NEGATIVE_SEED,
    _DEFAULT_K_NEIGHBORS,
    _DEFAULT_MODEL_OUT,
    _DEFAULT_RECORD_FRAMES,
    _DYNAMIC_TRAINING_SCOPE,
    _STATIC_TRAINING_SCOPE,
    _dynamic_metadata_out_for_type,
    _dynamic_model_out_for_type,
)


class _DummyPage:
    def run_thread(self, func, *args):
        return func(*args)


class _DummyController:
    def __init__(self):
        self.recording_calls = []
        self.training_calls = []
        self.negative_generation_calls = []

    def list_recorded_gestures(self):
        return []

    def start_recording(self, **kwargs):
        self.recording_calls.append(kwargs)
        return True

    def start_training(self, **kwargs):
        self.training_calls.append(kwargs)
        return True

    def start_negative_generation(self, **kwargs):
        self.negative_generation_calls.append(kwargs)
        return True

    def cancel_training(self):
        pass


def test_user_training_tab_uses_project_defaults():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._user_rec_label.value = "Wave"
    view._user_rec_samples.value = "7"
    view._user_rec_two_hands.value = True
    view._rec_frames.value = "99"

    view._on_record_start(None, mode="user")

    recording_call = controller.recording_calls[-1]
    assert recording_call["label"] == "Wave"
    assert recording_call["num_samples"] == 7
    assert recording_call["frames"] == _DEFAULT_RECORD_FRAMES
    assert recording_call["two_hands"] is True
    assert recording_call["include_global_motion"] is False

    view._tr_data_root.value = "custom/data"
    view._tr_out_path.value = "custom/model.pkl"
    view._tr_neighbors.value = "11"

    view._on_train_start(None, mode="user")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == _DEFAULT_DATA_ROOT
    assert training_call["out_path"] == _DEFAULT_MODEL_OUT
    assert training_call["neighbors"] == _DEFAULT_K_NEIGHBORS
    assert training_call["expect_dim"] == 42
    assert training_call["model_type"] == "knn"
    assert training_call["training_scope"] == _STATIC_TRAINING_SCOPE


def test_developer_training_tab_keeps_advanced_settings():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._rec_label.value = "DevWave"
    view._rec_samples.value = "3"
    view._rec_frames.value = "45"
    view._rec_two_hands.value = False

    view._on_record_start(None, mode="developer")

    recording_call = controller.recording_calls[-1]
    assert recording_call["label"] == "DevWave"
    assert recording_call["num_samples"] == 3
    assert recording_call["frames"] == 45
    assert recording_call["two_hands"] is False
    assert recording_call["include_global_motion"] is False

    view._tr_data_root.value = "dev/data"
    view._tr_out_path.value = "dev/model.pkl"
    view._tr_neighbors.value = "9"
    view._tr_model_type.value = "extra_trees"

    view._on_train_start(None, mode="developer")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == "dev/data"
    assert training_call["out_path"] == "dev/model.pkl"
    assert training_call["neighbors"] == 9
    assert training_call["expect_dim"] == 42
    assert training_call["model_type"] == "extra_trees"
    assert training_call["training_scope"] == _STATIC_TRAINING_SCOPE


def test_developer_dynamic_flow_uses_long_recording_and_separate_model():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_rec_label.value = "swipe_right"
    view._dyn_rec_samples.value = ""
    view._dyn_rec_frames.value = ""
    view._dyn_rec_two_hands.value = True

    view._on_record_start(None, mode="dynamic")

    recording_call = controller.recording_calls[-1]
    assert recording_call["label"] == "swipe_right"
    assert recording_call["num_samples"] == _DEFAULT_DYNAMIC_RECORD_SAMPLES
    assert recording_call["frames"] == _DEFAULT_DYNAMIC_RECORD_FRAMES
    assert recording_call["two_hands"] is True
    assert recording_call["include_global_motion"] is True

    view._dyn_model_out.value = ""
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_model_type.value = "sequence_mlp"
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == _DEFAULT_DATA_ROOT
    assert training_call["out_path"] == _DEFAULT_DYNAMIC_MODEL_OUT
    assert training_call["neighbors"] == _DEFAULT_K_NEIGHBORS
    assert training_call["expect_dim"] is None
    assert training_call["model_type"] == _DEFAULT_DYNAMIC_MODEL_TYPE
    assert training_call["feature_mode"] == _DEFAULT_DYNAMIC_FEATURE_MODE
    assert training_call["classes_out_path"] == "models/dynamic_sequence_mlp_classes.json"
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_sequence_mlp_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_sequence_mlp_feature_mode.txt"
    )
    assert training_call["training_scope"] == _DYNAMIC_TRAINING_SCOPE


def test_developer_negative_flow_generates_samples_automatically():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._neg_samples_per_label.value = ""
    view._neg_seed.value = ""

    view._on_negative_generate(None)

    assert controller.recording_calls == []
    generation_call = controller.negative_generation_calls[-1]
    assert generation_call["samples_per_label"] == _DEFAULT_NEGATIVE_SAMPLES_PER_LABEL
    assert generation_call["seed"] == _DEFAULT_NEGATIVE_SEED


def test_dynamic_model_type_keeps_production_sequence_mlp_output_path():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)

    view._dyn_model_type.value = _DEFAULT_DYNAMIC_MODEL_TYPE
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._on_dynamic_model_type_changed(None)

    assert view._dyn_model_out.value == _DEFAULT_DYNAMIC_MODEL_OUT
    assert view._dyn_feature_mode.value == "dynamic_sequence"

    view._dyn_model_type.value = "sequence_rocket"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._on_dynamic_model_type_changed(None)

    assert view._dyn_model_out.value == "models/dynamic_sequence_rocket.pkl"
    assert view._dyn_feature_mode.value == "dynamic_sequence"

    view._dyn_model_type.value = "sequence_shapelet_72"
    view._dyn_model_out.value = "models/dynamic_sequence_rocket.pkl"
    view._on_dynamic_model_type_changed(None)

    assert view._dyn_model_out.value == "models/dynamic_sequence_shapelet_72.pkl"
    assert view._dyn_feature_mode.value == "dynamic_sequence_72"


def test_developer_dynamic_flow_can_train_sequence_rocket_candidate():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_model_type.value = "sequence_rocket"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == "models/dynamic_sequence_rocket.pkl"
    assert training_call["model_type"] == "sequence_rocket"
    assert training_call["feature_mode"] == _DEFAULT_DYNAMIC_FEATURE_MODE
    assert training_call["classes_out_path"] == "models/dynamic_sequence_rocket_classes.json"
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_sequence_rocket_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_sequence_rocket_feature_mode.txt"
    )
    assert training_call["training_scope"] == _DYNAMIC_TRAINING_SCOPE


def test_legacy_dynamic_model_type_falls_back_to_sequence_mlp_paths():
    assert _dynamic_model_out_for_type("sequence_knn") == _DEFAULT_DYNAMIC_MODEL_OUT
    assert _dynamic_metadata_out_for_type("sequence_knn") == (
        "models/dynamic_sequence_mlp_classes.json",
        "models/dynamic_sequence_mlp_feature_dim.txt",
        "models/dynamic_sequence_mlp_feature_mode.txt",
    )


def test_sequence_mlp_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_mlp") == "models/dynamic_sequence_mlp.pkl"
    assert _dynamic_metadata_out_for_type("sequence_mlp") == (
        "models/dynamic_sequence_mlp_classes.json",
        "models/dynamic_sequence_mlp_feature_dim.txt",
        "models/dynamic_sequence_mlp_feature_mode.txt",
    )


def test_sequence_rocket_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_rocket") == (
        "models/dynamic_sequence_rocket.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_rocket") == (
        "models/dynamic_sequence_rocket_classes.json",
        "models/dynamic_sequence_rocket_feature_dim.txt",
        "models/dynamic_sequence_rocket_feature_mode.txt",
    )


def test_sequence_multirocket_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_multirocket") == (
        "models/dynamic_sequence_multirocket.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_multirocket") == (
        "models/dynamic_sequence_multirocket_classes.json",
        "models/dynamic_sequence_multirocket_feature_dim.txt",
        "models/dynamic_sequence_multirocket_feature_mode.txt",
    )


def test_sequence_sprocket_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_sprocket") == (
        "models/dynamic_sequence_sprocket.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_sprocket") == (
        "models/dynamic_sequence_sprocket_classes.json",
        "models/dynamic_sequence_sprocket_feature_dim.txt",
        "models/dynamic_sequence_sprocket_feature_mode.txt",
    )


def test_sequence_shapelet_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_shapelet") == (
        "models/dynamic_sequence_shapelet.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_shapelet") == (
        "models/dynamic_sequence_shapelet_classes.json",
        "models/dynamic_sequence_shapelet_feature_dim.txt",
        "models/dynamic_sequence_shapelet_feature_mode.txt",
    )


def test_sequence_shapelet_72_uses_long_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_shapelet_72") == (
        "models/dynamic_sequence_shapelet_72.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_shapelet_72") == (
        "models/dynamic_sequence_shapelet_72_classes.json",
        "models/dynamic_sequence_shapelet_72_feature_dim.txt",
        "models/dynamic_sequence_shapelet_72_feature_mode.txt",
    )


def test_developer_dynamic_flow_can_train_sequence_shapelet_72_candidate():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_model_type.value = "sequence_shapelet_72"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == "models/dynamic_sequence_shapelet_72.pkl"
    assert training_call["model_type"] == "sequence_shapelet"
    assert training_call["feature_mode"] == "dynamic_sequence_72"
    assert (
        training_call["classes_out_path"]
        == "models/dynamic_sequence_shapelet_72_classes.json"
    )
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_sequence_shapelet_72_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_sequence_shapelet_72_feature_mode.txt"
    )


def test_sequence_phase_hmm_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_phase_hmm") == (
        "models/dynamic_sequence_phase_hmm.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_phase_hmm") == (
        "models/dynamic_sequence_phase_hmm_classes.json",
        "models/dynamic_sequence_phase_hmm_feature_dim.txt",
        "models/dynamic_sequence_phase_hmm_feature_mode.txt",
    )


def test_sequence_ensemble_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_ensemble") == (
        "models/dynamic_sequence_ensemble.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_ensemble") == (
        "models/dynamic_sequence_ensemble_classes.json",
        "models/dynamic_sequence_ensemble_feature_dim.txt",
        "models/dynamic_sequence_ensemble_feature_mode.txt",
    )


def test_developer_dynamic_flow_can_train_sequence_ensemble_candidate():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_model_type.value = "sequence_ensemble"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == "models/dynamic_sequence_ensemble.pkl"
    assert training_call["model_type"] == "sequence_ensemble"
    assert training_call["feature_mode"] == "dynamic_sequence"
    assert (
        training_call["classes_out_path"]
        == "models/dynamic_sequence_ensemble_classes.json"
    )
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_sequence_ensemble_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_sequence_ensemble_feature_mode.txt"
    )
