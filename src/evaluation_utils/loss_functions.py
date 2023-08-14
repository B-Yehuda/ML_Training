from sklearn.metrics import mean_squared_error, average_precision_score, log_loss, \
    fbeta_score, d2_tweedie_score, r2_score, mean_absolute_error, mean_gamma_deviance, mean_tweedie_deviance


# LOSS FUNCTIONS #

class AveragePrecisionScore:
    def __init__(self, direction):
        self.direction = direction
        self.name = "Average_Precision_Score"

    def score(self, y_test, y_pred):
        return average_precision_score(y_test, y_pred)


class F1Score:
    def __init__(self, beta_value, direction):
        self.beta_value = beta_value
        self.direction = direction
        self.name = "F1_Score"

    def score(self, y_test, y_pred):
        return fbeta_score(y_test, y_pred, beta=self.beta_value)


class RecallScore:
    def __init__(self, beta_value, direction):
        self.beta_value = beta_value
        self.direction = direction
        self.name = "Recall_Score"

    def score(self, y_test, y_pred):
        return fbeta_score(y_test, y_pred, beta=self.beta_value)


class PrecisionScore:
    def __init__(self, beta_value, direction):
        self.beta_value = beta_value
        self.direction = direction
        self.name = "Precision_Score"

    def score(self, y_test, y_pred):
        return fbeta_score(y_test, y_pred, beta=self.beta_value)


class LogLoss:
    def __init__(self, direction):
        self.direction = direction
        self.name = "Log_Loss"

    def score(self, y_test, y_pred):
        return log_loss(y_test, y_pred)


class RMSE:
    def __init__(self, squared, direction):
        self.squared = squared
        self.direction = direction
        self.name = "Mean_Squared_Error"

    def score(self, y_test, y_pred):
        return mean_squared_error(y_test, y_pred, squared=self.squared)


class D2TweedieScore:
    def __init__(self, power, direction):
        self.power = power
        self.direction = direction
        self.name = "D2_Tweedie_Score"

    def score(self, y_test, y_pred):
        return d2_tweedie_score(y_test, y_pred, power=self.power)


class MeanTweedieScore:
    def __init__(self, power, direction):
        self.power = power
        self.direction = direction
        self.name = "Mean_Tweedie_Score"

    def score(self, y_test, y_pred):
        return mean_tweedie_deviance(y_test, y_pred, power=self.power)


class GammaScore:
    def __init__(self, direction):
        self.direction = direction
        self.name = "Gamma_Score"

    def score(self, y_test, y_pred):
        return mean_gamma_deviance(y_test, y_pred)


class R2:
    def __init__(self, direction):
        self.direction = direction
        self.name = "R2_Score"

    def score(self, y_test, y_pred):
        return r2_score(y_test, y_pred)


class MAE:
    def __init__(self, direction):
        self.direction = direction
        self.name = "Mean_Absolute_Error"

    def score(self, y_test, y_pred):
        return mean_absolute_error(y_test, y_pred)
