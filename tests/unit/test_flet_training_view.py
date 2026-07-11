from app.flet_app.views.training import (
    TrainingView,
    _DEFAULT_DATA_ROOT,
    _DEFAULT_DYNAMIC_FEATURE_MODE,
    _DEFAULT_DYNAMIC_MODEL_TYPE,
    _DEFAULT_DYNAMIC_MODEL_OUT,
    _DEFAULT_DYNAMIC_RECORD_FEATURE_DIM,
    _DEFAULT_DYNAMIC_RECORD_FRAMES,
    _DEFAULT_DYNAMIC_RECORD_SAMPLES,
    _DEFAULT_EXTRA_TREES_OPTUNA_CV_FOLDS,
    _DEFAULT_EXTRA_TREES_OPTUNA_TIMEOUT,
    _DEFAULT_EXTRA_TREES_OPTUNA_TRIALS,
    _DEFAULT_NEGATIVE_SAMPLES_PER_LABEL,
    _DEFAULT_NEGATIVE_SEED,
    _DEFAULT_K_NEIGHBORS,
    _DEFAULT_MODEL_OUT,
    _DEFAULT_MODEL_TYPE,
    _DEFAULT_RECORD_FRAMES,
    _DEFAULT_STATIC_FEATURE_MODE,
    _DEFAULT_STATIC_RECORD_FEATURE_DIM,
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
        self.recorded_gestures = []

    def list_recorded_gestures(self):
        return self.recorded_gestures

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


def _visible_text(control):
    parts = []
    seen = set()

    def visit(obj):
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, (str, int, float, bool)):
            parts.append(str(obj))
            return
        for attr in ("value", "label", "text", "tooltip", "key"):
            value = getattr(obj, attr, None)
            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))
        visit(getattr(obj, "content", None))
        for attr in ("controls", "options"):
            for child in getattr(obj, attr, []) or []:
                visit(child)

    visit(control)
    return "\n".join(parts)


def test_training_screen_hides_developer_dataset_log_and_protection_cards():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)

    text = _visible_text(view.build())

    assert "Пайплайн обучения" not in text
    assert "Менеджер датасета" not in text
    assert "Журнал" not in text
    assert "3. Защита" not in text
    assert "Состояние" in text


def test_dataset_manager_locks_non_user_classes():
    controller = _DummyController()
    controller.recorded_gestures = [
        {
            "label": "USER_WAVE",
            "samples": 2,
            "canDelete": True,
            "deleteReason": "",
        },
        {
            "label": "no_command",
            "samples": 20,
            "canDelete": False,
            "deleteReason": "Можно удалять только классы, записанные пользователем",
        },
    ]
    view = TrainingView(_DummyPage(), controller)

    view._refresh_datasets()
    view._request_delete_samples("no_command")

    assert "USER_WAVE" in view._dataset_can_delete_labels
    assert "no_command" not in view._dataset_can_delete_labels
    assert view._dataset_mix_text.value == "Базовые 1 · Свои 1"
    assert view._pending_delete_label == ""


def test_user_training_tab_uses_project_defaults():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._user_rec_label.value = "Wave"
    view._user_rec_samples.value = "7"
    view._user_rec_frames.value = "33"
    view._user_rec_two_hands.value = True
    view._rec_frames.value = "99"

    view._on_record_start(None, mode="user")

    recording_call = controller.recording_calls[-1]
    assert recording_call["label"] == "Wave"
    assert recording_call["num_samples"] == 7
    assert recording_call["frames"] == 33
    assert recording_call["two_hands"] is False
    assert recording_call["include_global_motion"] is False
    assert recording_call["include_landmark_z"] is True

    view._tr_data_root.value = "custom/data"
    view._tr_out_path.value = "custom/model.pkl"
    view._tr_neighbors.value = "11"

    view._on_train_start(None, mode="user")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == _DEFAULT_DATA_ROOT
    assert training_call["out_path"] == _DEFAULT_MODEL_OUT
    assert training_call["neighbors"] == _DEFAULT_K_NEIGHBORS
    assert training_call["expect_dim"] == _DEFAULT_STATIC_RECORD_FEATURE_DIM
    assert _DEFAULT_MODEL_TYPE == "extra_trees"
    assert training_call["model_type"] == "extra_trees"
    assert training_call["feature_mode"] == _DEFAULT_STATIC_FEATURE_MODE
    assert training_call["training_scope"] == _STATIC_TRAINING_SCOPE
    assert "--include-augmented" in training_call["extra_args"]
    assert "--extra-trees-optuna-trials" in training_call["extra_args"]
    trial_idx = training_call["extra_args"].index("--extra-trees-optuna-trials")
    assert training_call["extra_args"][trial_idx + 1] == str(
        _DEFAULT_EXTRA_TREES_OPTUNA_TRIALS
    )
    folds_idx = training_call["extra_args"].index("--extra-trees-optuna-cv-folds")
    assert training_call["extra_args"][folds_idx + 1] == str(
        _DEFAULT_EXTRA_TREES_OPTUNA_CV_FOLDS
    )
    timeout_idx = training_call["extra_args"].index("--extra-trees-optuna-timeout")
    assert training_call["extra_args"][timeout_idx + 1] == str(
        _DEFAULT_EXTRA_TREES_OPTUNA_TIMEOUT
    )


def test_user_recording_auto_trains_static_model_after_successful_save():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._user_rec_label.value = "Wave"

    view._on_record_start(None, mode="user")

    assert controller.training_calls == []
    controller.recording_calls[-1]["on_done"](0)

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == _DEFAULT_MODEL_OUT
    assert training_call["model_type"] == _DEFAULT_MODEL_TYPE
    assert training_call["feature_mode"] == _DEFAULT_STATIC_FEATURE_MODE
    assert training_call["training_scope"] == _STATIC_TRAINING_SCOPE


def test_user_recording_does_not_train_after_failed_save():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._user_rec_label.value = "Wave"

    view._on_record_start(None, mode="user")
    controller.recording_calls[-1]["on_done"](1)

    assert controller.training_calls == []


def test_user_dynamic_recording_auto_trains_dynamic_profile_from_start_state():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._user_rec_type.value = "dynamic"
    view._user_rec_label.value = "swipe_right"

    view._on_record_start(None, mode="user")
    view._user_rec_type.value = "static"
    controller.recording_calls[-1]["on_done"](0)

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == _DEFAULT_DYNAMIC_MODEL_OUT
    assert training_call["model_type"] == _DEFAULT_DYNAMIC_MODEL_TYPE
    assert training_call["feature_mode"] == _DEFAULT_DYNAMIC_FEATURE_MODE
    assert training_call["training_scope"] == _DYNAMIC_TRAINING_SCOPE


def test_unified_training_tab_handles_dynamic_gestures():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._user_rec_type.value = "dynamic"
    view._on_user_gesture_type_changed(None)
    view._user_rec_label.value = "swipe_right"
    view._user_rec_two_hands.value = True

    view._on_record_start(None, mode="user")

    recording_call = controller.recording_calls[-1]
    assert recording_call["label"] == "swipe_right"
    assert recording_call["num_samples"] == _DEFAULT_DYNAMIC_RECORD_SAMPLES
    assert recording_call["frames"] == _DEFAULT_DYNAMIC_RECORD_FRAMES
    assert recording_call["two_hands"] is False
    assert recording_call["include_global_motion"] is True
    assert recording_call["include_landmark_z"] is True

    view._on_train_start(None, mode="user")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == _DEFAULT_DATA_ROOT
    assert training_call["out_path"] == _DEFAULT_DYNAMIC_MODEL_OUT
    assert training_call["neighbors"] == _DEFAULT_K_NEIGHBORS
    assert training_call["expect_dim"] == _DEFAULT_DYNAMIC_RECORD_FEATURE_DIM
    assert training_call["model_type"] == _DEFAULT_DYNAMIC_MODEL_TYPE
    assert training_call["feature_mode"] == _DEFAULT_DYNAMIC_FEATURE_MODE
    assert training_call["training_scope"] == _DYNAMIC_TRAINING_SCOPE
    assert "--include-augmented" in training_call["extra_args"]


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
    assert recording_call["include_landmark_z"] is True

    view._tr_data_root.value = "dev/data"
    view._tr_out_path.value = "dev/model.pkl"
    view._tr_neighbors.value = "9"
    view._tr_model_type.value = "extra_trees"

    view._on_train_start(None, mode="developer")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == "dev/data"
    assert training_call["out_path"] == "dev/model.pkl"
    assert training_call["neighbors"] == 9
    assert training_call["expect_dim"] == _DEFAULT_STATIC_RECORD_FEATURE_DIM
    assert training_call["model_type"] == "extra_trees"
    assert training_call["feature_mode"] == _DEFAULT_STATIC_FEATURE_MODE
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
    assert recording_call["two_hands"] is False
    assert recording_call["include_global_motion"] is True
    assert recording_call["include_landmark_z"] is True

    view._dyn_model_out.value = ""
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_model_type.value = _DEFAULT_DYNAMIC_MODEL_TYPE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == _DEFAULT_DATA_ROOT
    assert training_call["out_path"] == _DEFAULT_DYNAMIC_MODEL_OUT
    assert training_call["neighbors"] == _DEFAULT_K_NEIGHBORS
    assert training_call["expect_dim"] is None
    assert training_call["model_type"] == _DEFAULT_DYNAMIC_MODEL_TYPE
    assert training_call["feature_mode"] == _DEFAULT_DYNAMIC_FEATURE_MODE
    assert (
        training_call["classes_out_path"]
        == "models/dynamic_landmark_lstm_backbone_classes.json"
    )
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_landmark_lstm_backbone_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_landmark_lstm_backbone_feature_mode.txt"
    )
    assert training_call["training_scope"] == _DYNAMIC_TRAINING_SCOPE
    assert "--sequence-lstm-optuna-trials" in training_call["extra_args"]


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


def test_dynamic_model_type_keeps_production_landmark_lstm_output_path():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)

    view._dyn_model_type.value = _DEFAULT_DYNAMIC_MODEL_TYPE
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._on_dynamic_model_type_changed(None)

    assert view._dyn_model_out.value == _DEFAULT_DYNAMIC_MODEL_OUT
    assert view._dyn_feature_mode.value == "dynamic_landmark_image"

    view._dyn_model_type.value = "sequence_mlp"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._on_dynamic_model_type_changed(None)

    assert view._dyn_model_out.value == "models/dynamic_sequence_mlp.pkl"
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
    assert training_call["feature_mode"] == "dynamic_sequence"
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


def test_legacy_dynamic_model_type_falls_back_to_production_landmark_lstm_paths():
    assert _dynamic_model_out_for_type("sequence_knn") == _DEFAULT_DYNAMIC_MODEL_OUT
    assert _dynamic_metadata_out_for_type("sequence_knn") == (
        "models/dynamic_landmark_lstm_backbone_classes.json",
        "models/dynamic_landmark_lstm_backbone_feature_dim.txt",
        "models/dynamic_landmark_lstm_backbone_feature_mode.txt",
    )


def test_dynamic_landmark_lstm_backbone_uses_own_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("dynamic_landmark_lstm_backbone") == (
        "models/dynamic_landmark_lstm_backbone.pkl"
    )
    assert _dynamic_metadata_out_for_type("dynamic_landmark_lstm_backbone") == (
        "models/dynamic_landmark_lstm_backbone_classes.json",
        "models/dynamic_landmark_lstm_backbone_feature_dim.txt",
        "models/dynamic_landmark_lstm_backbone_feature_mode.txt",
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


def test_sequence_gru_backbone_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_gru_backbone") == (
        "models/dynamic_sequence_gru_backbone.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_gru_backbone") == (
        "models/dynamic_sequence_gru_backbone_classes.json",
        "models/dynamic_sequence_gru_backbone_feature_dim.txt",
        "models/dynamic_sequence_gru_backbone_feature_mode.txt",
    )


def test_sequence_lstm_backbone_uses_own_sequence_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("sequence_lstm_backbone") == (
        "models/dynamic_sequence_lstm_backbone.pkl"
    )
    assert _dynamic_metadata_out_for_type("sequence_lstm_backbone") == (
        "models/dynamic_sequence_lstm_backbone_classes.json",
        "models/dynamic_sequence_lstm_backbone_feature_dim.txt",
        "models/dynamic_sequence_lstm_backbone_feature_mode.txt",
    )


def test_dynamic_landmark_cnn_uses_own_model_and_metadata_paths():
    assert _dynamic_model_out_for_type("dynamic_landmark_cnn") == (
        "models/dynamic_landmark_cnn.pkl"
    )
    assert _dynamic_metadata_out_for_type("dynamic_landmark_cnn") == (
        "models/dynamic_landmark_cnn_classes.json",
        "models/dynamic_landmark_cnn_feature_dim.txt",
        "models/dynamic_landmark_cnn_feature_mode.txt",
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


def test_developer_dynamic_flow_can_train_sequence_gru_backbone_candidate():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_model_type.value = "sequence_gru_backbone"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == "models/dynamic_sequence_gru_backbone.pkl"
    assert training_call["model_type"] == "sequence_gru_backbone"
    assert training_call["feature_mode"] == "dynamic_sequence"
    assert (
        training_call["classes_out_path"]
        == "models/dynamic_sequence_gru_backbone_classes.json"
    )
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_sequence_gru_backbone_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_sequence_gru_backbone_feature_mode.txt"
    )
    assert "--sequence-gru-optuna-trials" in training_call["extra_args"]


def test_developer_dynamic_flow_can_train_sequence_lstm_backbone_candidate():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_model_type.value = "sequence_lstm_backbone"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == "models/dynamic_sequence_lstm_backbone.pkl"
    assert training_call["model_type"] == "sequence_lstm_backbone"
    assert training_call["feature_mode"] == "dynamic_sequence"
    assert (
        training_call["classes_out_path"]
        == "models/dynamic_sequence_lstm_backbone_classes.json"
    )
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_sequence_lstm_backbone_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_sequence_lstm_backbone_feature_mode.txt"
    )
    assert "--sequence-lstm-optuna-trials" in training_call["extra_args"]


def test_developer_dynamic_flow_can_train_dynamic_landmark_lstm_backbone_candidate():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_model_type.value = "dynamic_landmark_lstm_backbone"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == "models/dynamic_landmark_lstm_backbone.pkl"
    assert training_call["model_type"] == "dynamic_landmark_lstm_backbone"
    assert training_call["feature_mode"] == "dynamic_landmark_image"
    assert (
        training_call["classes_out_path"]
        == "models/dynamic_landmark_lstm_backbone_classes.json"
    )
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_landmark_lstm_backbone_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_landmark_lstm_backbone_feature_mode.txt"
    )
    assert "--sequence-lstm-optuna-trials" in training_call["extra_args"]


def test_developer_dynamic_flow_can_train_dynamic_landmark_cnn_candidate():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._dyn_model_type.value = "dynamic_landmark_cnn"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._dyn_feature_mode.value = _DEFAULT_DYNAMIC_FEATURE_MODE
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["out_path"] == "models/dynamic_landmark_cnn.pkl"
    assert training_call["model_type"] == "dynamic_landmark_cnn"
    assert training_call["feature_mode"] == "dynamic_landmark_image"
    assert (
        training_call["classes_out_path"]
        == "models/dynamic_landmark_cnn_classes.json"
    )
    assert (
        training_call["feature_dim_out_path"]
        == "models/dynamic_landmark_cnn_feature_dim.txt"
    )
    assert (
        training_call["feature_mode_out_path"]
        == "models/dynamic_landmark_cnn_feature_mode.txt"
    )
