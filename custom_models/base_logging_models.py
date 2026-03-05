import abc
from typing import Optional
from transformers import PretrainedConfig


class Logger(abc.ABC):
    """
    Abstract base class for logging metrics.
    """

    def __init__(self, model=None):
        self.model = model
        self.config = None
        if model is not None:
            self.register_model(model)

    def register_model(self, model):
        if hasattr(model, "config"):
            self.config = model.config
        self.model = model

    @abc.abstractmethod
    def on_step_end(self, model_outputs):
        pass

    @abc.abstractmethod
    def on_log(self) -> dict:
        pass


class LoggingModel(abc.ABC):
    """
    Abstract base class for models that support logging.
    """

    def __init__(self, config, metrics_logger: Optional[Logger] = None):
        self.config = config
        self.metrics_logger = metrics_logger
        self.metrics_logger.register_model(self)

    def collect_and_store_metrics(self, model_outputs):
        if self.metrics_logger is not None:
            self.metrics_logger.on_step_end(model_outputs)

    def get_and_flush_metrics(self,) -> dict:
        stored_metrics = self.metrics_logger.on_log()
        return stored_metrics

    def forward(self, *args, **kwargs,):
        output = super().forward(*args, **kwargs)
        if self.metrics_logger is not None:
            self.collect_and_store_metrics(output)
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args,
                        metrics_logger: Optional[Logger] = None, **kwargs):
        if "metrics_logger" in kwargs:
            kwargs.pop("metrics_logger")
        cfg = kwargs.get("config", None)
        if isinstance(cfg, dict) and "metrics_logger" in cfg:
            cfg = {k: v for k, v in cfg.items() if k != "metrics_logger"}
            kwargs["config"] = cfg
        if isinstance(cfg, PretrainedConfig) and hasattr(cfg, "metrics_logger"):
            try:
                delattr(cfg, "metrics_logger")
            except Exception:
                pass

        model = super().from_pretrained(pretrained_model_name_or_path,
                                        *model_args, **kwargs)
        model.metrics_logger = metrics_logger
        if metrics_logger is not None:
            metrics_logger.register_model(model)
        return model
