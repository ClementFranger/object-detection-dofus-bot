import logging
import pygetwindow as gw
from kedro.io.core import AbstractDataset, DatasetError

logger = logging.getLogger(__name__)


class ApplicationDataset(AbstractDataset):

    def __init__(
            self,
            title: str = None,
    ):
        super().__init__()
        self.title = title

    def _window(self):
        windows = gw.getWindowsWithTitle(self.title)
        logger.info('Found {count} window for {title}'.format(count=len(windows), title=self.title))
        return next(iter(windows), None)

    def _describe(self):
        return {
            "title": self.title,
            "window": self._window()
        }

    def _load(self):
        return self._window()

    def _save(self, **kwargs):
        raise DatasetError(f"{self.__class__.__name__} is a read only data set type")

    def _exists(self):
        return self._window()
