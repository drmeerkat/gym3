import os
import copy
import cv2

from typing import Any, Mapping, Optional

import imageio
import numpy as np

from gym3.env import Env
from gym3.internal.renderer import Renderer, _str_to_array, _convert_ascii_to_rgba
from gym3.wrapper import Wrapper


class VideoRecorderWrapper(Wrapper):
    """
    Record observations of each episode from an environment to a video file

    Subclasses may want to override `_process_frame`

    :param env: environment to record from
    :param directory: directory to save videos to, will be created if it does not exist
    :param env_index: the index of the environment to record
    :param ob_key: by default the observation is recorded for the video, if the observation is a dictionary,
            you can specify which key to record using this argument
    :param info_key: if the frame you want to record is in the environment info dictionary, specify the key here, e.g. "rgb"
    :param prefix: filename prefix to use when creating videos
    :param fps: fps to give to encoder, this depends on your environment and the resulting
            video will play back too quickly or too slowly depending on this value
    :param writer_kwargs: extra arguments to supply to the imageio writer
    :param render: if set to True, also show the current frame being recorded in a window
    """

    def __init__(
        self,
        env: Env,
        directory: str,
        env_index: int = 0,
        ob_key: Optional[str] = None,
        info_key: Optional[str] = None,
        prefix: str = "",
        fps: int = 15,
        writer_kwargs: Optional[Mapping[str, Any]] = None,
        render=False,
        draw_overlay=True,
    ) -> None:
        super().__init__(env=env)
        if info_key is not None:
            assert ob_key is None, "can't specify both info_key and ob_key"
        self._prefix = prefix
        self._directory = os.path.abspath(directory)
        os.makedirs(self._directory, exist_ok=True)
        self._ob_key = ob_key
        self._info_key = info_key
        self._env_index = env_index
        self._episode_count = 0
        self._writer = None
        if writer_kwargs is None:
            writer_kwargs = {"output_params": ["-f", "mp4"]}
        self._writer_kwargs = writer_kwargs
        self._fps = fps
        self.videopath = None
        self._first_step = True
        self._renderer = Renderer(width=768, height=768) if render else None
        self._overlay_enabled = draw_overlay
        
        self._episode_steps = 0
        self._episode_return = 0

    def _restart_recording(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self.videopath = os.path.join(
            self._directory, f"{self._prefix}{self._episode_count:05d}.mp4"
        )
        self._writer = imageio.get_writer(
            self.videopath, format="ffmpeg", fps=self._fps, **self._writer_kwargs
        )
        # reset info counter
        self._episode_steps = 0
        self._episode_return = 0

    def _format_info(self) -> str:
        """
        Format the info for the current step into a string
        """
        info_rows = []
        for k, v in sorted(self._last_info.items()):
            info_rows.append(f"{k}: {v}")
        return "\n".join(info_rows)

    def _alpha_merge(self, small_foreground, background, top, left):
        """
        Puts a small BGRA picture in front of a larger BGR background.
        :param small_foreground: The overlay image. Must have 4 channels.
        :param background: The background. Must have 3 channels.
        :param top: Y position where to put the overlay.
        :param left: X position where to put the overlay.
        :return: a copy of the background with the overlay added.

        adopted from stackoverflow: https://stackoverflow.com/a/70579671/7463686
        """
        result = background.copy()
        # From everything I read so far, it seems we need the alpha channel separately
        # so let's split the overlay image into its individual channels
        fg_b, fg_g, fg_r, fg_a = cv2.split(small_foreground)
        # Make the range 0...1 instead of 0...255
        fg_a = fg_a / 255.0
        # Multiply the RGB channels with the alpha channel
        label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])

        # Work on a part of the background only
        height, width = small_foreground.shape[0], small_foreground.shape[1]
        part_of_bg = result[top:top + height, left:left + width, :]
        # Same procedure as before: split the individual channels
        bg_b, bg_g, bg_r = cv2.split(part_of_bg)
        # Merge them back with opposite of the alpha channel
        part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])

        # Add the label and the part of the background
        cv2.add(label_rgb, part_of_bg, part_of_bg)
        # Replace a part of the background
        result[top:top + height, left:left + width, :] = part_of_bg
        return result

    def _append_observation(self) -> None:
        rew, ob, _ = self.observe()
        # compute overlay info
        self._last_rew = rew[0]
        self._episode_return += self._last_rew
        self._episode_steps += 1
        info = copy.copy(self.env.get_info()[0])
        for k in list(info.keys()):
            if isinstance(info[k], np.ndarray):
                del info[k]
            if 'level' in k:
                del info[k]
        self._last_info = dict(
            episode_steps=self._episode_steps,
            episode_return=round(self._episode_return, 2),
            **info,
        )
        # render mode
        if self._info_key is None:
            if self._ob_key is not None:
                ob = ob[self._ob_key]
            img = ob[self._env_index]
        else:
            info = self.get_info()
            img = info[self._env_index].get(self._info_key)
            # the first info for a converted environment may be empty
            if self._first_step and img is None:
                return
            # draw overlay to the img!
            if self._overlay_enabled:
                text = self._format_info()
                if len(text) > 0:
                    # draw overlay with pyplot upon img
                    arr = _str_to_array(text)
                    text_img = _convert_ascii_to_rgba(arr, size_px=16)
                    text_bg = np.zeros(text_img.shape, dtype=np.uint8)
                    text_bg[:,:,3] = 130
                    # place the text img at the bottom left
                    img = self._alpha_merge(text_bg, img, img.shape[0] - (text_img.shape[0] + 10), 10)
                    img = self._alpha_merge(text_img, img, img.shape[0] - (text_img.shape[0] + 10), 10)
        frame = self._process_frame(img.astype(np.uint8))
        self._writer.append_data(frame)
        if self._renderer is not None:
            self._renderer.start()
            self._renderer.draw_bitmap(
                0, 0, self._renderer.width, self._renderer.height, image=frame
            )
            # supports overlay drawing to the map!
            if self._overlay_enabled:
                text = self._format_info()
                if len(text) > 0:
                    self._renderer.draw_text(
                        10,
                        10,
                        text=self._format_info(),
                        bg_alpha=0.5,
                        size_px=16,
                    )
            self._renderer.finish()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def act(self, ac: Any) -> None:
        if self._first_step:
            # first action of the episode, get the existing observation before
            # taking an action
            self._restart_recording()
            self._append_observation()

        super().act(ac)
        self._first_step = False
        _, _, first = self.observe()
        # generate frame img
        if first[self._env_index]:
            self._episode_count += 1
            self._first_step = True
            # self._append_observation() # test: will this add the last frame?
            self._writer.close()
            self._writer = None
        else:
            self._append_observation()
