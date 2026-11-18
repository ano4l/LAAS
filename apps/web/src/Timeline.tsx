import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { AnalysisResult } from "./main";
import type { Review } from "./ReviewPanel";
import { clock, clockTenths, words } from "./format";
import { useKits, type Tone } from "./kits";
import { usePlayback } from "./playback";
import { Swatch } from "./ui";
import { candidateName } from "./VideoInsights";
import { KIND_FILL, STATE_LABEL, STATE_TECHNICAL, segmentKind, type SegmentKind } from "./vocabulary";
import type { SequenceState } from "./TemporalPossession";

type Segment = { start: number; end: number; kind: SegmentKind; state: SequenceState; team: string | null };
type Span = { start: number; end: number };
type Marker = { key: string; at: number; shape: "touch" | "possible" | "reviewed" | "draft"; text: string; tone: Tone | null };

const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];

/**
 * The match timeline: three lanes under the video, sharing its clock.
 * Possession (colour = team, texture = certainty), ball seen, and markers. Click or drag to seek.
 */
export function Timeline({ result, saved }: { result: AnalysisResult; saved: Review | null }) {
  const playback = usePlayback();
  const kits = useKits();
  const duration = Math.max(result.metadata.duration_seconds, 0.001);
  const insights = result.metrics.video_insights;
  const sequences = insights?.sequences;
  const track = useRef<HTMLDivElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const readout = useRef<HTMLSpanElement>(null);
  const dragging = useRef(false);
  const [hover, setHover] = useState<{ time: number; x: number; width: number } | null>(null);

  const segments = useMemo<Segment[]>(() => (sequences?.intervals ?? []).map(interval => ({
    start: interval.start_seconds,
    end: interval.end_seconds,
    kind: segmentKind(interval.state, interval.team),
    state: interval.state,
    team: interval.team,
  })), [sequences]);
  const hasPossession = segments.some(segment => segment.kind !== "unknown");

  // Ball seen: runs of observed path points. A gap longer than the sampling step ends a run.
  const ballSpans = useMemo<Span[]>(() => {
    const path = insights?.ball.path ?? [];
    if (!path.length) return [];
    const fps = insights?.sample_fps ?? result.detection_summary?.sample_fps ?? result.metadata.fps;
    const step = fps > 0 ? Math.max(1, insights?.ball.path_display_stride ?? 1) / fps : 0.04;
    const points = [...path].sort((a, b) => a.timestamp_seconds - b.timestamp_seconds);
    const spans: Span[] = [];
    let start = points[0].timestamp_seconds;
    let previous = points[0];
    for (const point of points.slice(1)) {
      if (point.timestamp_seconds - previous.timestamp_seconds > step * 1.5 + 1e-6 || point.scene_id !== previous.scene_id) {
        spans.push({ start, end: Math.min(duration, previous.timestamp_seconds + step) });
        start = point.timestamp_seconds;
      }
      previous = point;
    }
    spans.push({ start, end: Math.min(duration, previous.timestamp_seconds + step) });
    return spans;
  }, [insights, result.detection_summary?.sample_fps, result.metadata.fps, duration]);

  const markers = useMemo<Marker[]>(() => {
    const list: Marker[] = [];
    sequences?.anchors.forEach(anchor => list.push({
      key: `touch-${anchor.id}`,
      at: anchor.start_seconds,
      shape: "touch",
      tone: kits.tone(anchor.team ?? anchor.kit),
      text: `Touch · ${kits.label(anchor.team ?? anchor.kit, "unknown group")} · track ${anchor.track_id}`,
    }));
    insights?.candidates.forEach(candidate => list.push({
      key: `possible-${candidate.id}`,
      at: candidate.timestamp_seconds,
      shape: "possible",
      tone: null,
      text: `${candidateName(candidate.type)} · to review`,
    }));
    saved?.events.forEach(event => list.push({
      key: `review-${event.id}`,
      at: event.timestamp_seconds,
      shape: event.review_status === "approved" ? "reviewed" : "draft",
      tone: null,
      text: `${words(event.type)} · ${saved.teams[event.team] || event.team} · ${event.review_status === "approved" ? "approved by a reviewer" : "saved draft, not approved"}`,
    }));
    return list.sort((a, b) => a.at - b.at);
  }, [sequences, insights, saved, kits]);

  const ticks = useMemo(() => {
    const step = TICK_STEPS.find(candidate => duration / candidate <= 8) ?? TICK_STEPS[TICK_STEPS.length - 1];
    const list: number[] = [];
    for (let value = 0; value <= duration + 1e-6; value += step) list.push(value);
    return list;
  }, [duration]);

  const teams = useMemo(() => [...new Set(segments.filter(segment => segment.team && (segment.kind === "supported" || segment.kind === "assumed")).map(segment => segment.team as string))].sort((a, b) => kits.label(a).localeCompare(kits.label(b))), [segments, kits]);
  const kinds = useMemo(() => new Set(segments.map(segment => segment.kind)), [segments]);
  const shapes = useMemo(() => new Set(markers.map(marker => marker.shape)), [markers]);

  const describe = (seconds: number) => {
    const segment = segments.find(item => seconds >= item.start && seconds < item.end);
    const state = !sequences ? "No possession estimate" : !segment || segment.kind === "unknown" ? STATE_LABEL.unknown : STATE_LABEL[segment.state];
    const team = segment && segment.team && segment.kind !== "unknown" ? kits.label(segment.team) : null;
    const technical = segment && segment.kind !== "unknown" ? STATE_TECHNICAL[segment.state] : null;
    const ball = !insights ? null : ballSpans.some(span => seconds >= span.start && seconds < span.end);
    return { segment, state, team, technical, ball };
  };

  // The playhead moves every video frame, so it is positioned directly rather than through React state.
  useEffect(() => {
    let lastSecond = -1;
    const update = () => {
      const seconds = Math.min(playback.getTime(), duration);
      if (head.current) head.current.style.left = `${(100 * seconds) / duration}%`;
      if (readout.current) readout.current.textContent = clockTenths(seconds);
      const whole = Math.floor(seconds);
      if (whole !== lastSecond && track.current) {
        lastSecond = whole;
        const now = describe(seconds);
        track.current.setAttribute("aria-valuenow", String(whole));
        track.current.setAttribute("aria-valuetext", `${clock(seconds)} of ${clock(duration)}. ${now.state}${now.team ? `, ${now.team}` : ""}${now.ball === null ? "" : now.ball ? ", ball seen" : ", ball not seen"}`);
      }
    };
    update();
    return playback.subscribe(update);
    // describe() only reads memoised data listed here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playback, duration, segments, ballSpans, kits]);

  const timeAt = (clientX: number) => {
    const box = track.current!.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - box.left) / box.width));
    return { time: ratio * duration, x: ratio * box.width, width: box.width };
  };

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    playback.seek(timeAt(event.clientX).time, { play: false });
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    const point = timeAt(event.clientX);
    setHover(point);
    if (dragging.current) playback.seek(point.time, { play: false });
  }

  function onPointerEnd(event: PointerEvent<HTMLDivElement>) {
    dragging.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (event.type !== "pointerup" || event.pointerType !== "mouse") setHover(null);
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const step = event.shiftKey ? 5 : 1;
    if (event.key === "ArrowRight" || event.key === "ArrowUp") playback.nudge(step);
    else if (event.key === "ArrowLeft" || event.key === "ArrowDown") playback.nudge(-step);
    else if (event.key === "Home") playback.seek(0, { play: false });
    else if (event.key === "End") playback.seek(duration, { play: false });
    else if (event.key === " " || event.key === "Enter") playback.toggle();
    else return;
    event.preventDefault();
  }

  const left = (seconds: number) => `${(100 * Math.max(0, Math.min(seconds, duration))) / duration}%`;
  const width = (start: number, end: number) => `${(100 * Math.max(0, Math.min(end, duration) - Math.max(0, start))) / duration}%`;
  const tip = hover ? describe(hover.time) : null;
  const nearMarkers = hover ? markers.filter(marker => Math.abs((marker.at / duration) * hover.width - hover.x) <= 6) : [];

  return (
    <section className="tl" aria-label="Match timeline">
      <header className="tl__head">
        <h2 className="tl__title">Match timeline</h2>
        <span className="tl__readout"><span ref={readout}>0:00.0</span> / {clock(duration)}</span>
      </header>

      <div className="tl__grid">
        <div className="tl__labels" aria-hidden="true">
          <span>Possession</span>
          <span>Ball seen</span>
          <span>Markers</span>
        </div>

        <div
          className="tl__track"
          ref={track}
          role="slider"
          tabIndex={0}
          aria-label="Match timeline. Left and right arrows move one second, with Shift five seconds."
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-orientation="horizontal"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerEnd}
          onPointerCancel={onPointerEnd}
          onPointerLeave={event => { if (!dragging.current) onPointerEnd(event); }}
          onKeyDown={onKeyDown}
        >
          <div className="tl__lane tl__lane--possession">
            {segments.filter(segment => segment.kind !== "unknown").map(segment => {
              const tone = kits.tone(segment.team);
              return <i key={`${segment.start}-${segment.state}`} className={`tl__seg fill-${KIND_FILL[segment.kind]} ${tone ? `tone-${tone}` : ""}`} style={{ left: left(segment.start), width: width(segment.start, segment.end) }} />;
            })}
            {!sequences && <span className="tl__empty">No possession estimate for this clip</span>}
            {sequences && !hasPossession && <span className="tl__empty">Possession unknown throughout this clip</span>}
          </div>

          <div className="tl__lane tl__lane--ball">
            {ballSpans.map(span => <i key={span.start} className="tl__seg fill-ball" style={{ left: left(span.start), width: width(span.start, span.end) }} />)}
            {!ballSpans.length && <span className="tl__empty">{insights ? "Ball not seen in this clip" : "No ball path for this clip"}</span>}
          </div>

          <div className="tl__lane tl__lane--marks">
            {markers.map(marker => <i key={marker.key} className={`tl__mark tl__mark--${marker.shape} ${marker.tone ? `tone-${marker.tone}` : ""}`} style={{ left: left(marker.at) }} />)}
            {!markers.length && <span className="tl__empty">No touches or events marked</span>}
          </div>

          <div className="tl__axis" aria-hidden="true">
            {ticks.map(value => <span key={value} className="tl__tick" style={{ left: left(value) }}><b>{clock(value)}</b></span>)}
          </div>

          {hover && <div className="tl__hoverline" style={{ left: hover.x }} aria-hidden="true" />}
          <div className="tl__playhead" ref={head} aria-hidden="true" />

          {hover && tip && (
            <div className="tl__tip" style={{ left: Math.max(96, Math.min(hover.width - 96, hover.x)) }} role="presentation">
              <b>{clockTenths(hover.time)}</b>
              <span title={tip.technical ?? undefined}>{tip.state}{tip.team ? ` · ${tip.team}` : ""}</span>
              {tip.ball !== null && <span className="muted">{tip.ball ? "Ball seen" : "Ball not seen"}</span>}
              {nearMarkers.slice(0, 3).map(marker => <span key={marker.key} className="tl__tipmark">{marker.text}</span>)}
            </div>
          )}
        </div>
      </div>

      <ul className="keyline tl__key" aria-label="Timeline key">
        {teams.map(team => <li key={team}><Swatch fill="solid" tone={kits.tone(team)} />{kits.label(team)}</li>)}
        {sequences && <li title="Estimated: on the ball, ball travelling or a brief gap"><Swatch fill="solid" />Supported</li>}
        {sequences && <li title="Assumed: kept between two touches by the same team"><Swatch fill="hatch" />Assumed</li>}
        {sequences && <li><Swatch fill="unknown" />Unknown</li>}
        {kinds.has("contested") && <li><Swatch fill="contested" />Contested</li>}
        {kinds.has("stoppage") && <li><Swatch fill="stoppage" />Stoppage</li>}
        {shapes.has("touch") && <li><i className="tl__mark tl__mark--touch tl__mark--key" aria-hidden="true" />Touch</li>}
        {shapes.has("possible") && <li><i className="tl__mark tl__mark--possible tl__mark--key" aria-hidden="true" />Possible event</li>}
        {shapes.has("reviewed") && <li><i className="tl__mark tl__mark--reviewed tl__mark--key" aria-hidden="true" />Reviewed event</li>}
        {shapes.has("draft") && <li><i className="tl__mark tl__mark--draft tl__mark--key" aria-hidden="true" />Saved draft</li>}
      </ul>
    </section>
  );
}
