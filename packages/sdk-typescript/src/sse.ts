// An incremental decoder for the server sent events wire format.
//
// One decoder serves every transport (decision D6): the fetch client feeds it
// bytes from a ReadableStream, the Angular transport feeds it the text that
// arrives with each XHR download progress event. Chunks may split anywhere, in
// the middle of a line, of a CRLF pair or of a multi byte character, so the
// decoder buffers until a blank line completes an event.

import type { AskEvent } from './types.js';
import { ASK_EVENT_NAMES } from './types.js';

export interface SseMessage {
  /** The event name; `message` when the block had no `event:` line, per the specification. */
  event: string;
  data: string;
}

export class SseDecoder {
  private buffer = '';
  private readonly text = new TextDecoder();
  private eventName = '';
  private dataLines: string[] = [];

  /** Feed a chunk and return every event it completed. */
  push(chunk: Uint8Array | string): SseMessage[] {
    this.buffer += typeof chunk === 'string' ? chunk : this.text.decode(chunk, { stream: true });
    return this.drain(false);
  }

  /** Flush at the end of the stream: a final event without a trailing blank line still counts. */
  end(): SseMessage[] {
    this.buffer += this.text.decode();
    const out = this.drain(true);
    if (this.dataLines.length > 0) {
      out.push(this.dispatch());
    }
    return out;
  }

  private drain(final: boolean): SseMessage[] {
    const out: SseMessage[] = [];
    for (;;) {
      const match = /\r\n|\n|\r/.exec(this.buffer);
      if (match === null) {
        if (final && this.buffer !== '') {
          this.line(this.buffer, out);
          this.buffer = '';
        }
        return out;
      }
      // A lone CR at the very end may be the first half of a CRLF still in flight.
      if (match[0] === '\r' && match.index === this.buffer.length - 1 && !final) {
        return out;
      }
      const line = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      this.line(line, out);
    }
  }

  private line(line: string, out: SseMessage[]): void {
    if (line === '') {
      if (this.dataLines.length > 0) {
        out.push(this.dispatch());
      } else {
        this.eventName = '';
      }
      return;
    }
    if (line.startsWith(':')) {
      return;
    }
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) {
      value = value.slice(1);
    }
    if (field === 'event') {
      this.eventName = value;
    } else if (field === 'data') {
      this.dataLines.push(value);
    }
    // `id` and `retry` are part of the format but mean nothing to this API.
  }

  private dispatch(): SseMessage {
    const message = { event: this.eventName || 'message', data: this.dataLines.join('\n') };
    this.eventName = '';
    this.dataLines = [];
    return message;
  }
}

const KNOWN = new Set<string>(ASK_EVENT_NAMES);

/** Parse one decoded message from /api/ask into a typed event. */
export function toAskEvent(message: SseMessage): AskEvent {
  const data: unknown = JSON.parse(message.data);
  if (KNOWN.has(message.event)) {
    return { event: message.event, data } as AskEvent;
  }
  return { event: 'unknown', name: message.event, data };
}
