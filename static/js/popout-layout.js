/* Popout layout — binary-split tree + operations.
 *
 * The popout window renders transcript/slides/terminal inside a tree of
 * nested splits. Leaves pick a panel by id; splits describe a direction
 * (horizontal or vertical) and the first child's share of the space.
 *
 *   type PanelId = 'transcript' | 'slides' | 'terminal';
 *   type Leaf    = { kind:'leaf',  panel: PanelId };
 *   type Split   = { kind:'split', dir:'h'|'v', ratio:number, a:Node, b:Node, id:string };
 *   type Node    = Leaf | Split;
 *
 * Split ids are namespaced by preset slug ("fullstack:main", "fullstack:bottom")
 * so ratio overrides persisted per-preset don't leak across presets.
 */

(function () {
  'use strict';

  const PANELS = ['transcript', 'slides', 'terminal'];
  const RATIO_MIN = 0.15;
  const RATIO_MAX = 0.85;

  function clampRatio(r) {
    if (!Number.isFinite(r)) return 0.5;
    if (r < RATIO_MIN) return RATIO_MIN;
    if (r > RATIO_MAX) return RATIO_MAX;
    return r;
  }

  function leaf(panel) {
    return { kind: 'leaf', panel };
  }

  function split(dir, ratio, a, b, id) {
    return { kind: 'split', dir, ratio: clampRatio(ratio), a, b, id };
  }

  function isLeaf(node) { return node && node.kind === 'leaf'; }
  function isSplit(node) { return node && node.kind === 'split'; }

  function walk(node, fn) {
    fn(node);
    if (isSplit(node)) {
      walk(node.a, fn);
      walk(node.b, fn);
    }
  }

  function collectPanels(tree) {
    const seen = new Set();
    walk(tree, (n) => {
      if (isLeaf(n)) seen.add(n.panel);
    });
    return Array.from(seen);
  }

  function collectSplitIds(tree) {
    const ids = [];
    walk(tree, (n) => {
      if (isSplit(n)) ids.push(n.id);
    });
    return ids;
  }

  /** Return a new tree with split `id`'s ratio set (clamped). Non-destructive. */
  function setRatio(tree, id, ratio) {
    if (isLeaf(tree)) return tree;
    if (tree.id === id) {
      return { ...tree, ratio: clampRatio(ratio) };
    }
    return {
      ...tree,
      a: setRatio(tree.a, id, ratio),
      b: setRatio(tree.b, id, ratio),
    };
  }

  function getRatio(tree, id) {
    if (isLeaf(tree)) return null;
    if (tree.id === id) return tree.ratio;
    const fromA = getRatio(tree.a, id);
    if (fromA !== null) return fromA;
    return getRatio(tree.b, id);
  }

  /** Apply per-preset ratio overrides to a tree. Missing ids are ignored. */
  function applyRatios(tree, overrides) {
    if (!overrides) return tree;
    let out = tree;
    for (const [id, ratio] of Object.entries(overrides)) {
      out = setRatio(out, id, ratio);
    }
    return out;
  }

  /** Validate a preset's tree: structure, panel ids, split-id uniqueness + slug prefix. */
  function validatePreset(tree, slug) {
    const errors = [];
    if (!tree) { errors.push('preset tree is null'); return errors; }

    const seenSplits = new Set();
    walk(tree, (n) => {
      if (!n || (n.kind !== 'leaf' && n.kind !== 'split')) {
        errors.push(`invalid node kind: ${JSON.stringify(n)}`);
        return;
      }
      if (isLeaf(n)) {
        if (!PANELS.includes(n.panel)) errors.push(`unknown panel id: ${n.panel}`);
        return;
      }
      // split
      if (n.dir !== 'h' && n.dir !== 'v') errors.push(`bad split dir: ${n.dir}`);
      if (typeof n.id !== 'string' || !n.id) errors.push('split missing id');
      if (n.id && !n.id.startsWith(`${slug}:`) && !n.id.startsWith('mobile:')) {
        errors.push(`split id '${n.id}' not namespaced with preset slug '${slug}:'`);
      }
      if (seenSplits.has(n.id)) errors.push(`duplicate split id within preset: ${n.id}`);
      seenSplits.add(n.id);
      if (!Number.isFinite(n.ratio) || n.ratio < 0 || n.ratio > 1) {
        errors.push(`split ${n.id} bad ratio: ${n.ratio}`);
      }
      if (!n.a || !n.b) errors.push(`split ${n.id} missing child(ren)`);
    });
    return errors;
  }

  /** Prune unavailable panels + promote siblings. Returns a Leaf or Split. */
  function resolveRenderableTree(tree, availability) {
    function prune(node) {
      if (isLeaf(node)) {
        return availability[node.panel] ? node : null;
      }
      const a = prune(node.a);
      const b = prune(node.b);
      if (a && b) return { ...node, a, b };
      return a || b;   // sibling promoted
    }
    const resolved = prune(tree);
    return resolved; // may be Leaf, Split, or null
  }

  function deepClone(node) {
    if (isLeaf(node)) return { ...node };
    return { ...node, a: deepClone(node.a), b: deepClone(node.b) };
  }

  // ── Tree mutations (v2 edit menu + DnD) ──────────────────────

  let _idCounter = 0;
  function _nextSplitId(slug) {
    _idCounter += 1;
    return `${slug}:custom-${Date.now().toString(36)}-${_idCounter}`;
  }

  /** Walk tree and apply `fn(parent, key, node)` where key is 'a'|'b'|null. */
  function walkPaths(tree, fn, parent = null, key = null) {
    fn(parent, key, tree);
    if (isSplit(tree)) {
      walkPaths(tree.a, fn, tree, 'a');
      walkPaths(tree.b, fn, tree, 'b');
    }
  }

  /** Return a new tree with the leaf at `targetPath` replaced by a split
   *  whose `side` (a or b) holds a new leaf of `newPanel`, and the
   *  other side keeps the original leaf. Non-destructive.
   */
  function splitAt(tree, targetPath, newPanel, dir, side, slug) {
    // Walk the path and rebuild.
    function go(node, path) {
      if (path.length === 0) {
        // This is the target leaf. Wrap it in a new split.
        if (!isLeaf(node)) return node;    // refuse to split a split
        const original = { kind: 'leaf', panel: node.panel };
        const fresh    = { kind: 'leaf', panel: newPanel };
        const a = side === 'a' ? fresh : original;
        const b = side === 'a' ? original : fresh;
        return { kind: 'split', dir, ratio: 0.5, a, b, id: _nextSplitId(slug || 'custom') };
      }
      if (!isSplit(node)) return node;
      const [head, ...rest] = path;
      return { ...node, [head]: go(node[head], rest) };
    }
    return go(tree, targetPath);
  }

  /** Remove the leaf at `targetPath`. Promotes the sibling up. Returns
   *  null if removing would empty the tree.
   */
  function removeAt(tree, targetPath) {
    if (targetPath.length === 0) return null;
    function go(node, path) {
      if (!isSplit(node)) return node;
      if (path.length === 1) {
        // The leaf to remove is node[path[0]]; return the sibling.
        const siblingKey = path[0] === 'a' ? 'b' : 'a';
        return node[siblingKey];
      }
      const [head, ...rest] = path;
      return { ...node, [head]: go(node[head], rest) };
    }
    return go(tree, targetPath);
  }

  /** Change the panel id of the leaf at `targetPath`. */
  function changePanelAt(tree, targetPath, panelId) {
    function go(node, path) {
      if (path.length === 0) {
        if (!isLeaf(node)) return node;
        return { ...node, panel: panelId };
      }
      if (!isSplit(node)) return node;
      const [head, ...rest] = path;
      return { ...node, [head]: go(node[head], rest) };
    }
    return go(tree, targetPath);
  }

  /** Swap two panels at paths pathA and pathB. Paths must both point to leaves. */
  function swapAt(tree, pathA, pathB) {
    const leafA = _getAt(tree, pathA);
    const leafB = _getAt(tree, pathB);
    if (!isLeaf(leafA) || !isLeaf(leafB)) return tree;
    let out = _setAt(tree, pathA, leafB);
    out = _setAt(out, pathB, leafA);
    return out;
  }

  function _getAt(tree, path) {
    if (path.length === 0) return tree;
    if (!isSplit(tree)) return null;
    const [head, ...rest] = path;
    return _getAt(tree[head], rest);
  }

  function _setAt(tree, path, newNode) {
    if (path.length === 0) return newNode;
    if (!isSplit(tree)) return tree;
    const [head, ...rest] = path;
    return { ...tree, [head]: _setAt(tree[head], rest, newNode) };
  }

  /** Find the DOM-walk path to the first leaf matching `panelId`. */
  function findPanelPath(tree, panelId) {
    let found = null;
    function go(node, path) {
      if (found) return;
      if (isLeaf(node)) {
        if (node.panel === panelId) found = path;
        return;
      }
      go(node.a, [...path, 'a']);
      go(node.b, [...path, 'b']);
    }
    go(tree, []);
    return found;
  }

  /** Build a fresh tree containing just one panel. */
  function singleLeaf(panelId) { return { kind: 'leaf', panel: panelId }; }

  window.PopoutLayout = {
    PANELS,
    RATIO_MIN,
    RATIO_MAX,
    leaf,
    split,
    isLeaf,
    isSplit,
    clampRatio,
    collectPanels,
    collectSplitIds,
    setRatio,
    getRatio,
    applyRatios,
    validatePreset,
    resolveRenderableTree,
    deepClone,
    // v2 edit operations
    splitAt,
    removeAt,
    changePanelAt,
    swapAt,
    findPanelPath,
    singleLeaf,
    walkPaths,
    _nextSplitId,
  };
})();
