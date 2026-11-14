import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useSyncExternalStore, type ReactNode } from "react";

/**
 * One video, shared by every panel. Any "watch" action seeks this player, and any panel can read the
 * current time to mark "now". Time lives outside React state: it changes every frame while playing,
 * so consumers subscribe through usePlaybackValue and re-render only when their derived value changes.
 */
export type SeekOptions = { play?: boolean };

export type Playback = {
  duration: number;
  getTime: () => number;
  isPlaying: () => boolean;
  subscribe: (listener: () => void) => () => void;
  /** Jump to a moment. "Watch" actions play from there; scrubbing passes { play: false }. */
  seek: (seconds: number, options?: SeekOptions) => void;
  /** Move relative to the current time without changing play state. */
  nudge: (deltaSeconds: number) => void;
  toggle: () => void;
  pause: () => void;
  /** Ref callback for the one <video> element. */
  attach: (element: HTMLVideoElement | null) => void | (() => void);
};

const PlaybackContext = createContext<Playback | null>(null);

export function PlaybackProvider({ duration, clipKey, children }: { duration: number; clipKey: string; children: ReactNode }) {
  const video = useRef<HTMLVideoElement | null>(null);
  const time = useRef(0);
  const playing = useRef(false);
  const pending = useRef<{ seconds: number; play: boolean } | null>(null);
  const listeners = useRef(new Set<() => void>());
  const frame = useRef(0);

  const playback = useMemo<Playback>(() => {
    const emit = () => listeners.current.forEach(listener => listener());
    const read = () => {
      const element = video.current;
      if (!element) return;
      const nextPlaying = !element.paused && !element.ended;
      if (element.currentTime === time.current && nextPlaying === playing.current) return;
      time.current = element.currentTime;
      playing.current = nextPlaying;
      emit();
    };
    const loop = () => {
      read();
      frame.current = playing.current ? window.requestAnimationFrame(loop) : 0;
    };
    const start = () => {
      read();
      if (!frame.current && playing.current) frame.current = window.requestAnimationFrame(loop);
    };
    const play = (element: HTMLVideoElement) => { element.play().catch(() => { /* autoplay can be refused; the position is still set */ }); };

    const seek: Playback["seek"] = (seconds, options) => {
        const target = Math.max(0, Math.min(Number.isFinite(seconds) ? seconds : 0, duration));
        const shouldPlay = options?.play ?? true;
        const element = video.current;
        time.current = target;
        if (element && element.readyState >= 1) {
          element.currentTime = target;
          pending.current = null;
          if (shouldPlay) play(element);
          // The player is sticky; this only matters if it has been scrolled away on a short screen.
          const box = element.getBoundingClientRect();
          if (box.bottom < 40 || box.top > window.innerHeight - 40) element.scrollIntoView({ block: "nearest", behavior: "smooth" });
        } else {
          pending.current = { seconds: target, play: shouldPlay };
        }
        emit();
    };

    return {
      duration,
      getTime: () => time.current,
      isPlaying: () => playing.current,
      subscribe(listener) {
        listeners.current.add(listener);
        return () => { listeners.current.delete(listener); };
      },
      seek,
      nudge(deltaSeconds) {
        seek((video.current?.currentTime ?? time.current) + deltaSeconds, { play: false });
      },
      toggle() {
        const element = video.current;
        if (!element) return;
        if (element.paused) play(element); else element.pause();
      },
      pause() { video.current?.pause(); },
      attach(element) {
        video.current = element;
        if (!element) return;
        const onMetadata = () => {
          const wanted = pending.current;
          if (wanted) {
            element.currentTime = wanted.seconds;
            pending.current = null;
            if (wanted.play) play(element);
          }
          read();
        };
        const events = ["timeupdate", "seeking", "seeked", "pause", "ended", "emptied"] as const;
        events.forEach(name => element.addEventListener(name, read));
        element.addEventListener("play", start);
        element.addEventListener("playing", start);
        element.addEventListener("loadedmetadata", onMetadata);
        if (element.readyState >= 1) onMetadata();
        return () => {
          events.forEach(name => element.removeEventListener(name, read));
          element.removeEventListener("play", start);
          element.removeEventListener("playing", start);
          element.removeEventListener("loadedmetadata", onMetadata);
          if (frame.current) window.cancelAnimationFrame(frame.current);
          frame.current = 0;
          if (video.current === element) video.current = null;
        };
      },
    };
  }, [duration]);

  // A different clip starts from the beginning; a seek aimed at the previous clip must not carry over.
  useEffect(() => {
    pending.current = null;
    time.current = 0;
    playing.current = false;
    listeners.current.forEach(listener => listener());
  }, [clipKey]);

  return <PlaybackContext.Provider value={playback}>{children}</PlaybackContext.Provider>;
}

export function usePlayback(): Playback {
  const playback = useContext(PlaybackContext);
  if (!playback) throw new Error("usePlayback must be used inside a PlaybackProvider");
  return playback;
}

/**
 * Subscribe to a value derived from the playhead. Return a primitive (an index, a rounded time, a
 * boolean) so the component re-renders only when that value changes, not on every video frame.
 */
export function usePlaybackValue<T extends string | number | boolean | null>(select: (seconds: number, playing: boolean) => T): T {
  const playback = usePlayback();
  const getSnapshot = useCallback(() => select(playback.getTime(), playback.isPlaying()), [playback, select]);
  return useSyncExternalStore(playback.subscribe, getSnapshot);
}
