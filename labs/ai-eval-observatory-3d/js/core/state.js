import { DEFAULT_STATE } from './config.js';

class ReactiveStore {
  constructor(initial) {
    this._state = { ...initial };
    this._listeners = new Map();
  }

  get(key) {
    return key ? this._state[key] : { ...this._state };
  }

  set(partial) {
    const changed = [];
    for (const [k, v] of Object.entries(partial)) {
      if (this._state[k] !== v) {
        this._state[k] = v;
        changed.push(k);
      }
    }
    if (changed.length === 0) return;

    for (const key of changed) {
      const fns = this._listeners.get(key);
      if (fns) fns.forEach(fn => fn(this._state[key], this._state));
    }
    const wildcard = this._listeners.get('*');
    if (wildcard) wildcard.forEach(fn => fn(changed, this._state));
  }

  subscribe(key, fn) {
    if (!this._listeners.has(key)) this._listeners.set(key, new Set());
    this._listeners.get(key).add(fn);
    return () => this._listeners.get(key)?.delete(fn);
  }
}

export const store = new ReactiveStore(DEFAULT_STATE);
