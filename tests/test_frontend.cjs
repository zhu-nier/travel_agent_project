// Browser-independent checks for message safety, socket state and tool correlation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const nodes = new Map();
class Element {
  constructor() { this.children = []; this.listeners = {}; this.value = ''; this.hidden = false; this.disabled = true; }
  set id(value) { this._id = value; nodes.set(value, this); }
  get id() { return this._id; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  prepend(child) { this.children.unshift(child); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute() {}
  focus() {}
  remove() { if (this.id) nodes.delete(this.id); }
  querySelector(selector) { return this.selectors?.[selector]; }
}
for (const id of ['messages', 'input', 'sendBtn', 'toolLog', 'emptyState', 'status', 'resetBtn', 'retryBtn']) {
  const element = new Element(); element.id = id;
}
nodes.get('emptyState').selectors = { p: new Element(), '.connecting': new Element() };
const sockets = [];
class Socket {
  static OPEN = 1;
  constructor(url) { this.url = url; this.readyState = 1; this.sent = []; sockets.push(this); }
  send(data) { this.sent.push(JSON.parse(data)); }
  close() { this.readyState = 3; }
  emit(data) { this.onmessage({ data: JSON.stringify(data) }); }
}
const timers = [];
const context = vm.createContext({
  document: { getElementById: id => nodes.get(id), createElement: () => new Element() },
  location: { protocol: 'https:', host: 'example.test' }, WebSocket: Socket,
  window: { addEventListener() {} },
  setTimeout: callback => { timers.push(callback); return timers.length; }, clearTimeout() {},
});
const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script).runInContext(context);
const first = sockets[0];
assert.equal(first.url, 'wss://example.test/ws');
assert.equal(nodes.get('input').disabled, true);
first.emit({ type: 'ready' });
assert.equal(nodes.get('input').disabled, false);
nodes.get('input').value = '<img src=x onerror=alert(1)>';
nodes.get('sendBtn').listeners.click();
assert.equal(first.sent.length, 1);
assert.equal(nodes.get('emptyState').hidden, true);
assert.equal(nodes.get('messages').children[0].children[0].textContent, '<img src=x onerror=alert(1)>');
assert.equal(nodes.get('messages').children[0].children[0].innerHTML, undefined);
nodes.get('input').value = 'duplicate';
nodes.get('sendBtn').listeners.click();
assert.equal(first.sent.length, 1, 'busy turns cannot be submitted twice');
first.emit({ type: 'tool_call', call_id: 'a', tool: 'weather', args: { city: '<script>' } });
first.emit({ type: 'tool_call', call_id: 'b', tool: 'weather', args: { city: '上海' } });
first.emit({ type: 'tool_result', call_id: 'a', result: 'first result' });
assert.equal(nodes.get('toolLog').children[1].children[2].textContent, 'first result');
assert.equal(nodes.get('toolLog').children[0].children[2].textContent, '执行中…');
first.emit({ type: 'turn_done' });
assert.equal(nodes.get('input').disabled, false);
nodes.get('resetBtn').listeners.click();
assert.equal(first.sent[1].type, 'reset');
first.emit({ type: 'reset_done' });
assert.equal(nodes.get('toolLog').children.length, 0);
first.readyState = 3;
first.onclose({ code: 1006 });
assert.equal(nodes.get('input').disabled, true);
assert.equal(timers.length, 1);
timers[0]();
const second = sockets[1];
second.emit({ type: 'ready' });
assert.match(nodes.get('messages').children.at(-1).children[0].textContent, /新对话/);
second.readyState = 3;
second.onclose({ code: 1011 });
assert.equal(timers.length, 1, 'configuration failures must not cause retry loops');
assert.equal(nodes.get('retryBtn').hidden, false);
console.log('Frontend checks passed: safe text, HTTPS socket, busy gate, tool IDs, reset and reconnect.');
