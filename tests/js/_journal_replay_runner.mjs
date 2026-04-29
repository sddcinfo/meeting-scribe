// Replay a scrubbed journal through the production SegmentStore and
// emit a JSON digest to stdout. Driven by tests/test_journal_replay.py.
//
// Usage: node tests/js/_journal_replay_runner.mjs <path-to-jsonl>
//
// Output (stdout, JSON):
//   {
//     "count": <int>,
//     "order": ["<segment_id>", ...],
//     "segments": { "<segment_id>": <store-record>, ... }
//   }

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { SegmentStore } from '../../static/js/segment-store.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const fixturePath = process.argv[2];
if (!fixturePath) {
  console.error('usage: node _journal_replay_runner.mjs <path-to-jsonl>');
  process.exit(2);
}

const lines = readFileSync(fixturePath, 'utf-8').split('\n');
const store = new SegmentStore();

for (const line of lines) {
  if (!line.trim()) continue;
  try {
    const event = JSON.parse(line);
    store.ingest(event);
  } catch (e) {
    console.error(`bad line in ${fixturePath}: ${e.message}`);
    process.exit(3);
  }
}

const segments = {};
for (const [sid, rec] of store.segments) {
  segments[sid] = rec;
}

const digest = {
  count: store.count,
  order: store.order,
  segments,
};

process.stdout.write(JSON.stringify(digest));
