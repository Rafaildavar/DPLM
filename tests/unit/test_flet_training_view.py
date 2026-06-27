from app.flet_app.views.training import (
    TrainingView,
    _DEFAULT_DATA_ROOT,
    _DEFAULT_DYNAMIC_CLASSES_OUT,
    _DEFAULT_DYNAMIC_FEATURE_DIM_OUT,
    _DEFAULT_DYNAMIC_FEATURE_MODE,
    _DEFAULT_DYNAMIC_FEATURE_MODE_OUT,
    _DEFAULT_DYNAMIC_MODEL_OUT,
    _DEFAULT_DYNAMIC_RECORD_FRAMES,
    _DEFAULT_DYNAMIC_RECORD_SAMPLES,
    _DEFAULT_NEGATIVE_RECORD_FRAMES,
    _DEFAULT_NEGATIVE_RECORD_SAMPLES,
    _DEFAULT_K_NEIGHBORS,
    _DEFAULT_MODEL_OUT,
    _DEFAULT_RECORD_FRAMES,
    _DYNAMIC_TRAINING_SCOPE,
    _dynamic_model_out_for_type,
)


class _DummyPage:
    def run_thread(self, func, *args):
        return func(*args)


class _DummyController:
    def __init__(self):
        self.recording_calls = []
        self.training_calls = []

    def list_recorded_gestures(self):
        return []

    def start_recording(self, **kwargs):
        self.recording_calls.append(kwargs)
        return True

    def start_training(self, **kwargs):
        self.training_calls.append(kwargs)
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
    assert training_call["model_type"] == "knn"


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
    assert training_call["model_type"] == "extra_trees"


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
    view._dyn_model_type.value = "svm"
    view._dyn_tr_neighbors.value = ""

    view._on_train_start(None, mode="dynamic")

    training_call = controller.training_calls[-1]
    assert training_call["data_root"] == _DEFAULT_DATA_ROOT
    assert training_call["out_path"] == _dynamic_model_out_for_type("svm")
    assert training_call["neighbors"] == _DEFAULT_K_NEIGHBORS
    assert training_call["model_type"] == "svm"
    assert training_call["feature_mode"] == _DEFAULT_DYNAMIC_FEATURE_MODE
    assert training_call["classes_out_path"] == _DEFAULT_DYNAMIC_CLASSES_OUT
    assert training_call["feature_dim_out_path"] == _DEFAULT_DYNAMIC_FEATURE_DIM_OUT
    assert training_call["feature_mode_out_path"] == _DEFAULT_DYNAMIC_FEATURE_MODE_OUT
    assert training_call["training_scope"] == _DYNAMIC_TRAINING_SCOPE


def test_developer_negative_flow_records_dynamic_negative_examples():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)
    view._neg_rec_label.value = "random_motion"
    view._neg_rec_samples.value = ""
    view._neg_rec_frames.value = ""
    view._neg_rec_two_hands.value = False

    view._on_record_start(None, mode="negative")

    recording_call = controller.recording_calls[-1]
    assert recording_call["label"] == "random_motion"
    assert recording_call["num_samples"] == _DEFAULT_NEGATIVE_RECORD_SAMPLES
    assert recording_call["frames"] == _DEFAULT_NEGATIVE_RECORD_FRAMES
    assert recording_call["two_hands"] is False
    assert recording_call["include_global_motion"] is True


def test_dynamic_model_type_updates_default_output_path_without_overriding_custom_path():
    controller = _DummyController()
    view = TrainingView(_DummyPage(), controller)

    view._dyn_model_type.value = "extra_trees"
    view._dyn_model_out.value = _DEFAULT_DYNAMIC_MODEL_OUT
    view._on_dynamic_model_type_changed(None)

    assert view._dyn_model_out.value == _dynamic_model_out_for_type("extra_trees")

    view._dyn_model_type.value = "svm"
    view._dyn_model_out.value = "models/custom_dynamic.pkl"
    view._on_dynamic_model_type_changed(None)

    assert view._dyn_model_out.value == "models/custom_dynamic.pkl"
