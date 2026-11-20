"""Offline viewer checks with Python stdlib and Node, without a browser service."""

from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import unittest


VIEWER = Path(__file__).with_name("review_viewer.html")


class ViewerElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.by_id = {}
        self.external_assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.by_id[attrs["id"]] = attrs
        for name in ("src", "href"):
            if attrs.get(name, "").startswith(("https://", "http://", "//")):
                self.external_assets.append(attrs[name])


class ReviewViewerTests(unittest.TestCase):
    def test_standalone_template_and_hidden_gt(self):
        source = VIEWER.read_text()
        parser = ViewerElements()
        parser.feed(source)
        self.assertEqual(source.count("__REVIEW_DATA__"), 1)
        self.assertFalse(parser.external_assets)
        self.assertNotIn("fetch(", source)
        self.assertNotIn("XMLHttpRequest", source)
        for element in ("gtDetails", "postGtFields", "gtLegend", "historyControls"):
            self.assertIn("hidden", parser.by_id[element])
        script = re.findall(r"<script>([\s\S]*?)</script>", source)[-1]
        referenced_ids = set(re.findall(r'\$\("([^\"]+)"\)', script))
        self.assertFalse(referenced_ids - parser.by_id.keys())

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the vanilla JS interaction checks")
    def test_js_blinding_history_label_roundtrip_and_package_isolation(self):
        harness = r'''
const fs = require("node:fs"), vm = require("node:vm"), assert = require("node:assert/strict");
const source = fs.readFileSync(process.argv[1], "utf8");
const script = [...source.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
class Element {
  constructor() {
    this.textContent = ""; this.children = []; this.dataset = {}; this.value = "";
    this.hidden = false; this.checked = false; this.naturalWidth = 800; this.naturalHeight = 600;
    this.attributes = {}; this.listeners = {}; this.classList = {toggle() {}};
  }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute(key,value) { this.attributes[key] = value; }
  addEventListener(event,callback) { this.listeners[event] = callback; }
  click() {} remove() {}
}
const elements = new Map();
const document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
  createElement() { return new Element(); }, createElementNS() { return new Element(); },
  querySelectorAll() { return []; }, addEventListener() {}, body:new Element(),
};
for (const match of source.matchAll(/\bid="([^"]+)"/g)) document.getElementById(match[1]);
function frame(offset) {
  return {frame_index:100+offset, relative_to_anchor:offset, image:"assets/frame.png", heatmap:"assets/heatmap.png",
    bbox_norm:[0.1,0.1,0.2,0.3], pred_xy:[0.4,0.5], score:0.6, inout:1,
    gazex_norm:[0.7], gazey_norm:[0.8], l2:0.42, path:"images/clip/"+(100+offset)+".jpg"};
}
const data = {schema_version:1, package_id:"package-a", title:"Do not reveal this descriptive title", protocol:{test_used_for_manual_analysis:true}, cases:[
  {case_id:"train_large_l2_secret", split:"train", category:"large_l2", track_id:"train/clip/person", anchor_gt_inout:1, anchor_l2:0.42,
    frames:[frame(-64),frame(-32),frame(-16),frame(-1),frame(0),frame(1)]},
  {case_id:"test_inout_fn_secret", split:"test", category:"inout_fn", track_id:"test/clip/person", anchor_gt_inout:1, anchor_l2:0.3,frames:[frame(0)]},
]};
document.getElementById("reviewData").textContent = JSON.stringify(data);
document.getElementById("splitFilter").value = "all";
document.getElementById("categoryFilter").value = "all";
document.getElementById("playSpeed").value = "500";
const storage = new Map(), timers = {};
const context = vm.createContext({document, console, setTimeout() {return 1;}, clearTimeout() {},
  setInterval(callback) {timers.tick=callback;return 2;}, clearInterval() {timers.tick=null;},
  localStorage:{getItem:key=>storage.get(key)||null, setItem:(key,value)=>storage.set(key,value)},
  URL:{createObjectURL() {return "blob:local";},revokeObjectURL() {}}, Blob:class {},
});
vm.runInContext(script, context);
const run = code => vm.runInContext(code, context);
assert.equal(run("gtVisible"), false);
assert.equal(run("historyVisible"), false);
assert.equal(run("historyWindow"), 16);
assert.equal(run("currentFrame().relative_to_anchor"), 0);
assert.equal(run("validFrames().length"), 1);
assert.equal(elements.get("gtDetails").hidden, true);
assert.equal(elements.get("postGtFields").hidden, true);
assert.equal(elements.get("gtCaseId").textContent, "");
assert.equal(elements.get("gtCategory").textContent, "");
assert.equal(elements.get("caseTitle").textContent, "CASE 001");
assert.equal(elements.get("caseList").children[0].children[1].textContent, "CASE 001");
run("setHistory(true)");
assert.equal(run("JSON.stringify(validFrames().map(f=>f.relative_to_anchor))"), "[-16,-1,0]");
run("historyWindow=64;setHistory(true)");
assert.equal(run("validFrames().length"), 5);
assert.equal(run("validFrames().some(f=>f.relative_to_anchor>0)"), false);
run("setFrame(0);togglePlayback()");
assert.equal(run("playTimer !== null"), true);
while(timers.tick) timers.tick();
assert.equal(run("currentFrame().relative_to_anchor"), 0);
run("toggleGt()");
const firstReveal = run("labelFor(selectedId).gt_revealed_at");
assert.ok(Number.isFinite(Date.parse(firstReveal)));
assert.equal(elements.get("gtDetails").hidden, false);
assert.equal(elements.get("gtCategory").textContent, "大 L2");
run("toggleGt()");
assert.equal(elements.get("gtDetails").hidden, true);
assert.equal(run("labelFor(selectedId).gt_revealed_at"), firstReveal);
run('labelFor(selectedId).current_guess="a,\\\"b\\\"\\n中文";persist()');
assert.equal(run("makeExport().labels.length"), 2);
assert.equal(run("makeExport().labels[1].current_guess"), "");
assert.equal(run("makeExport().package_id"), "package-a");
const before = run("JSON.stringify(labels)");
run('var wrong=makeExport();wrong.package_id="package-b"');
assert.throws(() => run("importLabels(wrong)"), /package_id/);
assert.equal(run("JSON.stringify(labels)"), before);
run('var unknown=makeExport();unknown.labels[0].case_id="unknown"');
assert.throws(() => run("importLabels(unknown)"), /case_id/);
assert.equal(run("JSON.stringify(labels)"), before);
run('var duplicate=makeExport();duplicate.labels[1].case_id=duplicate.labels[0].case_id');
assert.throws(() => run("importLabels(duplicate)"), /case_id/);
assert.equal(run("JSON.stringify(labels)"), before);
run('var emptyReveal=makeExport();emptyReveal.labels[0].gt_revealed_at="";importLabels(emptyReveal)');
assert.equal(run("labelFor(selectedId).gt_revealed_at"), firstReveal);
assert.equal(run('csvCell("a,b\\\"c")'), '"a,b""c"');
run('selectCase("test_inout_fn_secret")');
assert.equal(run("gtVisible"), false);
assert.equal(run("historyVisible"), false);
assert.equal(elements.get("caseTitle").textContent, "CASE 002");
assert.equal(run('categoryName("inout_false_positive")'), "InOut 假阳性");
assert.equal(run('categoryName("inout_false_negative")'), "InOut 假阴性");
assert.equal(run('categoryName("medium_l2")'), "中等错误");
assert.equal(run('categoryName("correct_control")'), "正确对照");
run('$("splitFilter").value="test";$("categoryFilter").value="large_l2";applyFilters()');
assert.equal(elements.get("mainContent").hidden, true);
assert.equal(elements.get("notesPanel").hidden, true);
assert.equal(elements.get("emptyResults").hidden, false);
run('$("splitFilter").value="all";$("categoryFilter").value="all";applyFilters()');
assert.equal(elements.get("mainContent").hidden, false);
assert.equal(elements.get("notesPanel").hidden, false);
assert.equal(elements.get("emptyResults").hidden, true);
assert.ok(storage.has("cscwd-review-labels-v1:package-a"));
console.log("Offline viewer: blinding, causal windows, playback, full exports and import isolation passed.");
'''
        result = subprocess.run([shutil.which("node"), "-e", harness, str(VIEWER)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
