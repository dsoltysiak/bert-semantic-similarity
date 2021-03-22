import config
import tensorflow_addons as tfa
from tensorflow.keras import metrics
from encoding import BertPreprocessing


def test_eval(model, test_df, y_test):
    test_data = BertPreprocessing(
        test_df[["sentence1", "sentence2"]].values.astype("str"),
        y_test,
        batch_size=config.batch_size,
        shuffle=False,
    )

    y_pred = model.predict(test_data)

    size = y_pred.shape[0]
    y_test = y_test[:size, :]

    accuracy = metrics.CategoricalAccuracy()
    accuracy.update_state(y_test, y_pred)

    precision = metrics.Precision()
    precision.update_state(y_test, y_pred)

    recall = metrics.Recall()
    recall.update_state(y_test, y_pred)

    f1 = tfa.metrics.F1Score(num_classes=3, average="macro")
    f1.update_state(y_test, y_pred)

    auc = metrics.AUC()
    auc.update_state(y_test, y_pred)

	print(f"""
	Accuracy: {accuracy.result().numpy()}
	Precision: {precision.result().numpy()}
	Recall: {recall.result().numpy()}
	F1 score: {f1.result().numpy()}
	AUC: {auc.result().numpy()}
	""")

