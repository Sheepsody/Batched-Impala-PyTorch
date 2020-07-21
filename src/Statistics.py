from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from threading import Thread
from queue import Empty


class SummaryType(Enum):
    SCALAR = 1
    HISTOGRAM = 2
    VIDEO = 3
    IMAGE = 4
    FIGURE = 5
    GRAPH = 6


# Not working asynchronously
class Statistics(Thread):
    """Writes the statistics of the async processes into a tensorboard"""

    def __init__(self, writer_dir, statistics_queue, nb_episodes):

        super(Statistics, self).__init__()

        self.exit = False

        self.stats_queue = statistics_queue
        self.nb_episodes = nb_episodes

        self._writer = SummaryWriter(log_dir=writer_dir)

    def run(self):

        super(Statistics, self).run()

        # Make sure that all the logs are pushed to the tensorboard
        while True:

            try:
                summary_type, tag, data = self.stats_queue.get(timeout=1)
            except Empty:
                if self.exit:
                    break
                continue

            # Push the informations
            step = self.nb_episodes.value

            if summary_type == summary_type.SCALAR:
                self._writer.add_scalar(tag=tag, scalar_value=data, global_step=step)

            elif summary_type == summary_type.HISTOGRAM:
                self._writer.add_histogram(
                    tag=tag, values=data, global_step=step, bins="tensorflow"
                )

            elif summary_type == summary_type.FIGURE:
                self._writer.add_figure(tag=tag, figure=data, global_step=step)

            elif summary_type == summary_type.IMAGE:
                if data.dim() > 3:
                    self._writer.add_images(
                        tag=tag, img_tensor=data, global_step=step, dataformats="NCHW"
                    )
                else:
                    self._writer.add_image(
                        tag=tag, img_tensor=data, global_step=step, dataformats="CHW"
                    )

            elif summary_type == summary_type.VIDEO:
                self._writer.add_video(tag=tag, data=data, global_step=step, fps=4)

            elif summary_type == summary_type.GRAPH:
                self._writer.add_graph(model=data)

        self._writer.close()
