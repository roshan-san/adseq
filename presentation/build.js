// Project Phase II deck: RNN/GRU/LSTM vs lag-FNN for a full-scale anaerobic digester.
// node build.js results.json out.pptx   (results.json from export_results.py)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(
  "/home/roshanr/.claude/skills/synced/3c558126-52d2-421b-a994-e31d5054e06d_3331d544-f9bb-471e-8f73-dcad2c362388/pptx/scripts/apply_theme.js",
);

const [, , resPath, outPath] = process.argv;
const RES = JSON.parse(fs.readFileSync(resPath, "utf8"));
const NAMES = { persistence: "Persistence", fnn_paper: "Lag-FNN (paper)", fnn_tuned: "Lag-FNN (tuned)", window_fnn: "Window-FNN", rnn: "RNN", gru: "GRU", lstm: "LSTM" };
const SUM = Object.values(RES.summary)[0];
const pBetter = (m, vs) => (RES.sig.find((r) => r.model === m && r.vs === vs) || {}).p_model_better;
// recurrent models that beat BOTH FNN baselines (one-sided Diebold-Mariano, p < 0.05)
const WINS = RES.banner ? [] : ["rnn", "gru", "lstm"].filter((m) => pBetter(m, "fnn_paper") < 0.05 && pBetter(m, "fnn_tuned") < 0.05);
const TOP = WINS.length ? WINS.reduce((a, b) => (SUM[a].r2 >= SUM[b].r2 ? a : b)) : null;
const pct = (a, b) => Math.round((100 * (b - a)) / b);

const THEME = {
  name: "Digester",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A24", lt1: "FFFFFF", dk2: "1F4E3D", lt2: "EAF2EC",
    accent1: "2E7D5B", accent2: "E09F3E", accent3: "8DB580",
    accent4: "5B6C64", accent5: "B85042", accent6: "3A6EA5",
    hlink: "2E7D5B", folHlink: "5B6C64",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Project Phase II – Recurrent vs Lag-FNN forecasting of an anaerobic digester";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;
const M = 0.6; // side margin
const CW = W - 2 * M; // content width

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "tag", type: "body", x: M, y: 1.2, w: CW, h: 0.4, fontSize: 14, bold: true, color: C.accent2, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.7, w: 10.5, h: 2.0, fontSize: 40, bold: true, color: C.background1, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.8, w: 10.5, h: 0.6, fontSize: 18, color: C.accent3, margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "tag", type: "body", x: M, y: 0.35, w: CW, h: 0.35, fontSize: 12, bold: true, color: C.accent2, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.7, w: CW, h: 0.75, fontSize: 34, bold: true, color: C.text2, valign: "top", margin: 0 }, text: "" } },
    { text: { text: "Project Phase II  ·  Team [No.]", options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right", margin: 0 },
});

// ---------- helpers ----------
async function icon(Comp, hex, px = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: px }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function content(section, tag, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(tag, { placeholder: "tag" });
  s.addText(title, { placeholder: "title" });
  return s;
}
const T = (text, opts) => ({ text, options: opts || {} });
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: { type: "none" }, rectRadius: 0.12, objectName: name,
  });
}
async function iconBadge(s, Comp, x, y, d = 0.6, bg = C.accent1) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { type: "none" } });
  s.addImage({ data: await icon(Comp, "FFFFFF"), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56 });
}
function arrow(s, x, y, w, h, head = true) {
  s.addShape(pres.shapes.LINE, {
    x, y, w, h, line: { color: HEX.accent4, width: 1.75, endArrowType: head ? "triangle" : undefined },
  });
}

(async () => {
  // ===== Title =====
  pres.addSection({ title: "Title" });
  {
    const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Title" });
    s.addText("PROJECT PHASE II REVIEW", { placeholder: "tag" });
    s.addText("Recurrent Neural Networks vs. Lag-Feature FNN for Forecasting a Full-Scale Anaerobic Digester", { placeholder: "title" });
    s.addText("RNN · GRU · LSTM on public full-scale plant data (Muscatine WRRF, Iowa) — daily biogas", { placeholder: "body" });
    const rows = [
      ["Team No.", "[Team No.]"],
      ["Members", "Roshan Jeyaruban R (43731160),  Akash Kumar K (43731011)"],
      ["Guide", "Dr. M. Shanthi Thangam, Assistant Professor"],
      ["Department", "CSE with Artificial Intelligence, Sathyabama Institute of Science and Technology, Chennai"],
    ];
    s.addText(
      rows.flatMap(([k, v], i) => [
        T(k + ":  ", { bold: true, color: C.accent2 }),
        T(v, { color: C.background1, breakLine: i < rows.length - 1 }),
      ]),
      { x: M, y: 4.9, w: 10.5, h: 1.6, fontSize: 16, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true },
    );
    s.addText("17 November 2026", { x: M, y: 6.7, w: 4, h: 0.35, fontSize: 14, color: C.accent3, margin: 0, isTextBox: true });
    s.addNotes("Introduce the team and guide. One line: we forecast the daily biogas production of a real full-scale digester and test whether recurrent networks beat the feed-forward network of the base paper.");
  }

  // ===== 1. Abstract =====
  pres.addSection({ title: "1 Abstract" });
  {
    const s = content("1 Abstract", "01 · ABSTRACT", "Abstract");
    s.addText(
      [
        T("Full-scale anaerobic digesters (AD) produce unstable biogas because feedstock and operating conditions change every day. Long et al. (2026) forecast a commercial digester with a feed-forward network (FNN) on hand-made lag features of 1, 2, 7 and 15 days.", { breakLine: true }),
        T("This project replaces those fixed lags with recurrent networks — RNN, GRU and LSTM — that learn how much history matters directly from a 15-day input sequence.", { breakLine: true }),
        T("The base paper's data is not public, so we use the open daily dataset of the Muscatine, Iowa wastewater plant (2020–2023). All models share one fair protocol: a held-out final 12 months, leakage-free features, an equal W&B Sweeps tuning budget, 10 random seeds and Diebold–Mariano significance tests." + (TOP
          ? ` On the held-out year the ${NAMES[TOP]} reached R² ${SUM[TOP].r2.toFixed(2)} vs. ${SUM.fnn_paper.r2.toFixed(2)} for the base-paper FNN (RMSE −${pct(SUM[TOP].rmse, SUM.fnn_paper.rmse)} %, p ${pBetter(TOP, "fnn_paper") < 0.001 ? "< 0.001" : "= " + pBetter(TOP, "fnn_paper").toFixed(3)}).`
          : "")),
      ],
      { x: M, y: 1.65, w: 7.9, h: 5.0, fontSize: 16, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 12, isTextBox: true },
    );
    const stats = [["3", "years of daily full-scale plant data"], ["7", "models, same test days"], ["10", "random seeds per model"]];
    for (const [i, [big, small]] of stats.entries()) {
      const y = 1.65 + i * 1.7;
      card(s, 9.0, y, 3.73, 1.45, C.background2, "stat" + i);
      s.addText(big, { x: 9.25, y: y + 0.12, w: 1.2, h: 1.2, fontSize: 54, bold: true, fontFace: THEME.headFontFace, color: C.accent1, valign: "middle", margin: 0, isTextBox: true });
      s.addText(small, { x: 10.45, y: y + 0.12, w: 2.15, h: 1.2, fontSize: 15, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    }
    s.addNotes("Read the abstract in your own words. Stress: same data, same test period, same tuning budget for every model, so any gain comes from the model, not from extra tuning. The base paper's MSU data is only available on request, so we use a public full-scale plant, which also makes the study reproducible.");
  }

  // ===== 2. Problem statement =====
  pres.addSection({ title: "2 Problem Statement" });
  {
    const s = content("2 Problem Statement", "02 · PROBLEM STATEMENT", "Problem statement");
    card(s, M, 1.65, 5.6, 4.95, C.text2, "statement");
    s.addText(
      [
        T("Can recurrent neural networks that learn temporal memory from daily sequences forecast the biogas production of a full-scale digester more accurately than the lag-feature FNN — under a fair, leakage-free comparison?", { fontSize: 20, bold: true, color: C.background1, breakLine: true }),
        T(" ", { fontSize: 10, breakLine: true }),
        T("Plant: Muscatine, Iowa water resource recovery facility — two 1,840 m³ mesophilic CSTRs (36 °C) co-digesting sewage sludge, food/industrial waste and FOG; public daily data 2020–2023.", { fontSize: 15, color: C.accent3 }),
      ],
      { x: M + 0.35, y: 1.95, w: 4.9, h: 4.4, valign: "top", margin: 0, isTextBox: true },
    );
    const gaps = [
      [fa.FaHistory, "Memory fixed by hand", "Lags of 1, 2, 7 and 15 days are chosen manually, but a digester's memory (solids retention ≈ 3–4 weeks) is longer and varies."],
      [fa.FaProjectDiagram, "No state over time", "The FNN treats every day independently; it cannot carry the process state from one day to the next."],
      [fa.FaBalanceScale, "Comparison not yet rigorous", "Grid-searched single model, 80/20 split method unstated, no repeated seeds or significance tests reported."],
    ];
    for (const [i, [Ic, head, body]] of gaps.entries()) {
      const y = 1.65 + i * 1.7;
      card(s, 6.6, y, 6.13, 1.45, C.background2, "gap" + i);
      await iconBadge(s, Ic, 6.85, y + 0.4, 0.65, C.accent2);
      s.addText([T(head, { bold: true, fontSize: 17, color: C.text2, breakLine: true }), T(body, { fontSize: 14, color: C.text1 })],
        { x: 7.75, y: y + 0.1, w: 4.8, h: 1.25, valign: "middle", margin: 0, isTextBox: true });
    }
    s.addNotes("Three gaps motivate the project. 1) The base paper fixes memory by hand. 2) A feed-forward network has no internal state. 3) To claim one model beats another we need tuned baselines, repeated runs and a significance test.");
  }

  // ===== 3. Literature survey =====
  pres.addSection({ title: "3 Literature Survey" });
  {
    const s = content("3 Literature Survey", "03 · LITERATURE SURVEY", "Literature survey: prior work");
    const hdr = (t) => T(t, { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } });
    const rows = [
      ["De Clercq et al. (2020)", "Interpretable ML, industrial co-digestion plant", "Biomethane production predictable from plant records", "Tabular model; time handled by hand"],
      ["Kazemi et al. (2020)", "SVM, ELM, ensemble NN soft sensors", "Robust online VFA estimation", "Lab-scale data only"],
      ["Gupta et al. (2023)", "Review: explainable ML for AD", "ML + ADM1 gives faster, adaptive models", "Review; few full-scale studies"],
      ["Zhu, Yang & Ren (2023)", "Review: ML pitfalls in environmental research", "Random splits of time series leak information", "Motivates our chronological split"],
      ["Long et al. (2025)", "Supervised ML on the SCAD plant", "Commercial AD with diverse feedstocks is predictable", "Static models"],
      ["Long et al. (2026) — base paper", "128-64-1 ReLU FNN on lag + rolling features", "Test 2020: R² 0.67 biogas, 0.73 CH₄, 0.86 H₂S", "Fixed hand-made memory; no significance test"],
      ["Schroer & Just (2023) — our data", "Ridge, lasso, MLP, AutoML on Muscatine plant", "1-day-ahead biogas, MAPE ≈ 27 % (daily data)", "No recurrent models; single run"],
    ];
    s.addTable(
      [[hdr("Study"), hdr("Method"), hdr("Key finding"), hdr("Limitation / gap")],
        ...rows.map((r, i) => r.map((c, j) => T(c, {
          bold: j === 0, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 }, color: i >= 5 && j === 3 ? HEX.accent5 : HEX.dk1,
        })))],
      { x: M, y: 1.65, w: CW, colW: [2.75, 3.3, 3.4, 2.68], fontSize: 13, fontFace: THEME.bodyFontFace, valign: "middle", border: { type: "none" }, margin: [4, 8, 4, 8], rowH: 0.6 },
    );
    s.addNotes("Most AD machine-learning studies use static supervised models or lab data. Zhu et al. warn that random splits of time series leak information, which is why we split chronologically. The base paper is the strongest full-scale study; its weakness is the hand-made memory. Schroer and Just published the Muscatine dataset we use; their best daily model was an MLP and they did not try recurrent networks.");
  }
  {
    const s = content("3 Literature Survey", "03 · LITERATURE SURVEY", "Literature survey: methods used");
    const items = [
      [fa.FaSyncAlt, "Elman (1990)", "Simple recurrent network: a hidden state carries information from one time step to the next."],
      [fa.FaMemory, "Hochreiter & Schmidhuber (1997)", "LSTM: input, forget and output gates keep long-term memory and avoid vanishing gradients."],
      [fa.FaFilter, "Cho et al. (2014)", "GRU: update and reset gates; fewer parameters than LSTM with similar accuracy."],
      [fa.FaSlidersH, "Biewald (2020)", "Weights & Biases Sweeps: Bayesian hyper-parameter search; every run and learning curve logged."],
      [fa.FaChartLine, "Diebold & Mariano (1995); Newey & West (1987)", "DM test: are two forecasts' errors significantly different? HAC variance for autocorrelated daily errors."],
      [fa.FaShieldAlt, "Zhu et al. (2023)", "Best practice: past-only features, scaling fit on training data, chronological validation."],
    ];
    for (const [i, [Ic, head, body]] of items.entries()) {
      const col = i % 2, row = Math.floor(i / 2);
      const x = M + col * 6.215, y = 1.65 + row * 1.7;
      card(s, x, y, 5.9, 1.45, C.background2, "method" + i);
      await iconBadge(s, Ic, x + 0.25, y + 0.4, 0.65);
      s.addText([T(head, { bold: true, fontSize: 16, color: C.text2, breakLine: true }), T(body, { fontSize: 14, color: C.text1 })],
        { x: x + 1.1, y: y + 0.1, w: 4.6, h: 1.25, valign: "middle", margin: 0, isTextBox: true });
    }
    s.addNotes("These are the building blocks: three recurrent cells, the W&B Sweeps tuner, and the Diebold-Mariano test used to decide whether a difference in error is real.");
  }

  // ===== 4. System architecture =====
  pres.addSection({ title: "4 System Architecture" });
  {
    const s = content("4 System Architecture", "04 · SYSTEM ARCHITECTURE", "System architecture");
    const box = (x, y, w, h, fill, txt, head, name) => {
      card(s, x, y, w, h, fill, name);
      s.addText([T(head, { bold: true, fontSize: 15, color: txt, breakLine: true }), T(name, { fontSize: 12, color: txt })],
        { x: x + 0.15, y, w: w - 0.3, h, align: "center", valign: "middle", margin: 0, isTextBox: true });
    };
    // row A: data pipeline
    const aW = 2.75, aG = (CW - 4 * aW) / 3, aY = 1.6, aH = 1.15;
    const A = [
      ["Muscatine daily data", "2020–2023 · feeds, pH, temperature, VFA, alkalinity, biogas"],
      ["Pre-processing", "Daily calendar · causal forward-fill · labels never filled"],
      ["Feature engineering", "Lags 1/2/7/15 · rolling mean/std 5/10/15 · cyclic calendar"],
      ["Split & scaling", "Train | val (last 20 %) | test = last 12 months · min-max fit on train"],
    ];
    A.forEach(([h, d], i) => box(M + i * (aW + aG), aY, aW, aH, C.background2, C.text2, h, d));
    for (let i = 0; i < 3; i++) arrow(s, M + i * (aW + aG) + aW, aY + aH / 2, aG, 0);
    // row B: two input branches
    const bW = 5.9, bY = 3.25, bH = 1.2, bX = [M, W - M - bW];
    box(bX[0], bY, bW, bH, C.accent1, C.background1, "Tabular input  X  (N × F)", "Persistence · Lag-FNN (base paper) · Lag-FNN (tuned)");
    box(bX[1], bY, bW, bH, C.accent1, C.background1, "Sequence input  (N × 15 days × F)", "Window-FNN · RNN · GRU · LSTM");
    const splitCx = M + 3 * (aW + aG) + aW / 2, busA = 2.98;
    const bC = bX.map((x) => x + bW / 2);
    arrow(s, splitCx, aY + aH, 0, busA - aY - aH, false);
    arrow(s, bC[0], busA, splitCx - bC[0], 0, false);
    bC.forEach((cx) => arrow(s, cx, busA, 0, bY - busA));
    // row C: training, evaluation, outputs
    const cW = 3.8, cG = (CW - 3 * cW) / 2, cY = 5.05, cH = 1.25;
    const Cb = [
      ["Training + W&B Sweeps", "Adam · MSE · early stopping · 40 Bayesian runs per model"],
      ["Evaluation", "10 seeds · RMSE, MAE, R², ±10 % · Diebold–Mariano test"],
      ["Outputs", "IEEE results table · test-period prediction plots · W&B logs"],
    ];
    Cb.forEach(([h, d], i) => box(M + i * (cW + cG), cY, cW, cH, C.accent2, C.text1, h, d));
    for (let i = 0; i < 2; i++) arrow(s, M + i * (cW + cG) + cW, cY + cH / 2, cG, 0);
    const busB = 4.75, trainCx = M + cW / 2;
    bC.forEach((cx) => arrow(s, cx, bY + bH, 0, busB - bY - bH, false));
    arrow(s, trainCx, busB, bC[1] - trainCx, 0, false);
    arrow(s, trainCx, busB, 0, cY - busB);
    s.addNotes("Walk left to right: raw daily records are put on a calendar, features are built from past days only, the data is split by time and scaled using training data only. FNN models get one day's feature vector; recurrent models get the last 15 days as a sequence. Every model is trained by the same loop and tuned with the same W&B Sweeps budget, then scored with 10 seeds and the Diebold-Mariano test.");
  }

  // ===== 5. Updated work =====
  const UW = "5 Updated Work";
  const UWT = "05 · UPDATED WORK";
  pres.addSection({ title: UW });
  {
    const s = content(UW, UWT, "Work completed in Phase II");
    const items = [
      [fa.FaDatabase, "Public dataset + pipeline", "Public Muscatine plant data; daily calendar, causal gap filling, base-paper lag features"],
      [fa.FaShieldAlt, "Leakage-free protocol", "Past-only features, chronological split, scaling fit on training data; checked by unit tests"],
      [fa.FaNetworkWired, "Seven models", "Persistence, base-paper FNN, tuned FNN, Window-FNN, RNN, GRU and LSTM in PyTorch"],
      [fa.FaSlidersH, "Equal-budget tuning", "W&B Sweeps Bayesian search, 40 runs for every tuned model"],
      [fa.FaChartBar, "Statistical evaluation", "10 seeds, mean ± std, Diebold–Mariano tests against both FNN baselines"],
      [fa.FaTerminal, "Reproducible software", "One-command CLI (tyro), W&B logging, automatic LaTeX table and figures"],
    ];
    for (const [i, [Ic, head, body]] of items.entries()) {
      const col = i % 3, row = Math.floor(i / 3);
      const cw = (CW - 2 * 0.35) / 3, x = M + col * (cw + 0.35), y = 1.65 + row * 2.55;
      card(s, x, y, cw, 2.3, C.background2, "work" + i);
      await iconBadge(s, Ic, x + 0.3, y + 0.3, 0.7);
      s.addText([T(head, { bold: true, fontSize: 17, color: C.text2, breakLine: true }), T(body, { fontSize: 14, color: C.text1 })],
        { x: x + 0.3, y: y + 1.1, w: cw - 0.6, h: 1.1, valign: "top", margin: 0, isTextBox: true });
    }
    s.addNotes("Overview of everything built in Phase II. The base paper's MSU data is only available on request and did not arrive, so we moved to the public Muscatine dataset (Schroer & Just, University of Iowa). The next slides go through the protocol, the models, training and evaluation, and the first results.");
  }
  {
    const s = content(UW, UWT, "Leakage-free data protocol");
    const rules = [
      ["Calendar-daily index", "missing days become gaps, so “lag 7” is exactly 7 days ago"],
      ["Past-only target features", "lags and rolling statistics use days before t only"],
      ["Causal gap filling", "inputs carried forward up to 15 days; labels never imputed"],
      ["Train-only scaling", "min-max scaler fitted on the training period only"],
      ["Same test days for all", "every model needs a full 15-day history, so all are scored on identical days"],
    ];
    for (const [i, [h, d]] of rules.entries()) {
      const y = 1.65 + i * 0.95;
      s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.08, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { type: "none" } });
      s.addText(String(i + 1), { x: M, y: y + 0.08, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText([T(h, { bold: true, color: C.text2, breakLine: true }), T(d, { color: C.text1 })],
        { x: M + 0.8, y, w: 5.9, h: 0.8, fontSize: 15, valign: "middle", margin: 0, isTextBox: true });
    }
    // split timeline
    const tx = 7.7, tw = W - M - tx;
    s.addText("Chronological split", { x: tx, y: 1.65, w: tw, h: 0.45, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
    const nAll = RES.n.train + RES.n.val + RES.n.test;
    const segs = [
      ["Train", `May 2020 – Dec 2021 (${RES.n.train} d)`, RES.n.train / nAll, C.accent1, C.background1],
      ["Val", `${RES.n.val} d`, RES.n.val / nAll, C.accent3, C.text1],
      ["Test", `Apr 2022 – Mar 2023 (${RES.n.test} d)`, RES.n.test / nAll, C.accent2, C.text1],
    ];
    let cx = tx;
    for (const [lab, sub, frac, fill, txt] of segs) {
      const w = tw * frac;
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 2.3, w, h: 0.8, fill: { color: fill }, line: { color: HEX.lt1, width: 2 } });
      s.addText(lab, { x: cx, y: 2.3, w, h: 0.8, fontSize: 16, bold: true, color: txt, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(sub, { x: cx, y: 3.15, w, h: 0.6, fontSize: 12, color: C.accent4, align: "center", valign: "top", margin: 0, isTextBox: true });
      cx += w;
    }
    s.addText("→ time", { x: tx, y: 3.7, w: tw, h: 0.35, fontSize: 12, italic: true, color: C.accent4, align: "right", margin: 0, isTextBox: true });
    card(s, tx, 4.35, tw, 2.15, C.background2, "tests");
    s.addText(
      [T("Verified by automated tests", { bold: true, fontSize: 16, color: C.text2, breakLine: true }),
        T("Changing day t or later never changes day t's features", { bullet: true, breakLine: true }),
        T("Train < validation < test; test days never seen in training", { bullet: true, breakLine: true }),
        T("Last step of each window equals the FNN input", { bullet: true })],
      { x: tx + 0.3, y: 4.5, w: tw - 0.6, h: 1.9, fontSize: 14, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true },
    );
    s.addNotes("Leakage makes a model look better than it is. These five rules keep future information out. A random 80/20 split of daily data would put neighbouring days in train and validation; we split by time instead. The 2021 flow-meter outage (6 Apr – 27 Jun) is left as a gap, so no input window crosses it. Three pytest tests check the rules automatically.");
  }
  {
    const s = content(UW, UWT, "Models compared");
    const hdr = (t) => T(t, { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } });
    const rows = [
      ["Persistence", "yesterday's value", "Sanity floor: “tomorrow = today”", "—"],
      ["Lag-FNN (base paper)", "features of day t", "128-64-1 ReLU, Adam lr 1e-3, as published", "Fixed"],
      ["Lag-FNN (tuned)", "features of day t", "Same idea, tuned with the same budget", "40 runs"],
      ["Window-FNN", "15-day window, flattened", "Ablation: same input as RNNs, no recurrence", "40 runs"],
      ["RNN (Elman)", "15-day sequence", "Last hidden state → linear output", "40 runs"],
      ["GRU", "15-day sequence", "Update / reset gates", "40 runs"],
      ["LSTM", "15-day sequence", "Input / forget / output gates", "40 runs"],
    ];
    const prop = (i) => i >= 4;
    s.addTable(
      [[hdr("Model"), hdr("Input"), hdr("Role"), hdr("Tuning")],
        ...rows.map((r, i) => r.map((c, j) => T(c, { bold: j === 0, color: prop(i) && j === 0 ? HEX.accent1 : HEX.dk1, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } })))],
      { x: M, y: 1.65, w: 8.4, colW: [2.3, 2.2, 2.9, 1.0], fontSize: 14, fontFace: THEME.bodyFontFace, valign: "middle", border: { type: "none" }, margin: [5, 8, 5, 8], rowH: 0.58 },
    );
    card(s, 9.35, 1.65, W - M - 9.35, 4.64, C.background2, "why");
    s.addText(
      [T("Why these baselines?", { bold: true, fontSize: 17, color: C.text2, breakLine: true }),
        T("Beating only an untuned baseline is not convincing, so the FNN is run both as published and tuned.", { bullet: true, breakLine: true }),
        T("Window-FNN separates the effect of recurrence from the effect of simply seeing more days.", { bullet: true, breakLine: true }),
        T("A useful model must beat persistence.", { bullet: true })],
      { x: 9.6, y: 1.85, w: W - M - 9.85, h: 4.3, fontSize: 14, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 8, isTextBox: true },
    );
    s.addNotes("Seven models, all scored on identical test days. The green names are the proposed recurrent models. The two FNN baselines answer the reviewer question 'did you just tune your model harder?'. Window-FNN answers 'is it the recurrence or just the extra input?'.");
  }
  {
    const s = content(UW, UWT, "Training, tuning and evaluation");
    const cw = (CW - 2 * 0.35) / 3;
    const cols = [
      [fa.FaCogs, "Training (shared by all)", ["Adam optimiser, MSE loss", "Mini-batches, gradient clipping 1.0", "Early stopping on validation RMSE (patience 60)", "Best-epoch weights restored"]],
      [fa.FaSlidersH, "Tuning (W&B Sweeps)", ["40 runs per model: equal budget", "Bayesian search on validation RMSE", "Learning rate 1e-4 – 1e-2, weight decay, batch 16–128, dropout 0–0.4", "FNN: 1–3 layers, width 32–256; RNN: hidden 16–128, 1–2 layers"]],
      [fa.FaChartBar, "Evaluation", ["RMSE, MAE, R², ±10 % accuracy", "10 seeds → mean ± std", "Diebold–Mariano test (Newey–West HAC) vs. both FNN baselines", "Same test days (Apr 2022 – Mar 2023) for every model"]],
    ];
    for (const [i, [Ic, head, lines]] of cols.entries()) {
      const x = M + i * (cw + 0.35);
      card(s, x, 1.65, cw, 4.85, C.background2, "col" + i);
      await iconBadge(s, Ic, x + 0.3, 1.9, 0.7);
      s.addText(head, { x: x + 1.15, y: 1.9, w: cw - 1.35, h: 0.7, fontSize: 18, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
      s.addText(lines.map((l, k) => T(l, { bullet: true, breakLine: k < lines.length - 1 })),
        { x: x + 0.3, y: 2.85, w: cw - 0.6, h: 3.5, fontSize: 15, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 8, isTextBox: true });
    }
    s.addNotes("Every neural model goes through the same training loop. The tuned models each get exactly 40 W&B sweep runs, which keeps the comparison fair. Final scores are averaged over 10 seeds, and the Diebold-Mariano test tells us whether a lower error is statistically significant.");
  }
  {
    const s = content(UW, UWT, RES.banner ? "First results on the Muscatine data" : "Results on the Muscatine data");
    if (RES.banner) {
      s.addText(RES.banner, {
        x: M, y: 1.55, w: CW, h: 0.45, fontSize: 14, bold: true, color: C.background1, fill: { color: C.accent5 }, align: "center", valign: "middle", margin: 0, isTextBox: true,
      });
    }
    const sr = RES.series;
    s.addChart(pres.charts.LINE, [
      { name: "Measured", labels: sr.dates, values: sr.measured },
      { name: "Lag-FNN (base paper)", labels: sr.dates, values: sr.fnn_paper },
      { name: "RNN", labels: sr.dates, values: sr.rnn },
    ], {
      x: M, y: 2.15, w: 7.6, h: 4.45,
      showTitle: true, title: "Daily biogas, test period (Apr 2022 – Mar 2023): measured vs. predicted", titleFontSize: 14, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
      chartColors: [HEX.dk1, HEX.accent5, HEX.accent1], lineSize: 1.5, lineDataSymbol: "none",
      catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, catAxisLabelColor: HEX.accent4, valAxisLabelColor: HEX.accent4,
      catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFrequency: 20,
      valGridLine: { color: "D9E2DC", size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
      valAxisTitle: "ft³/min (daily mean)", showValAxisTitle: true, valAxisTitleFontSize: 10, valAxisTitleColor: HEX.accent4, valAxisTitleFontFace: "+mn-lt",
    });
    const order = ["persistence", "fnn_paper", "fnn_tuned", "window_fnn", "rnn", "gru", "lstm"];
    const res = RES.summary[sr.target];
    const best = order.filter((m) => res[m]).reduce((a, b) => (res[a].r2 >= res[b].r2 ? a : b));
    const hdr = (t) => T(t, { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center" });
    const cols = [["R²", "r2", 2], ["RMSE", "rmse", 1], ["MAE", "mae", 1]];
    s.addText("Test scores (mean of seeds)", { x: 8.55, y: 2.15, w: W - M - 8.55, h: 0.4, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addTable(
      [[hdr("Model"), ...cols.map(([h]) => hdr(h))],
        ...order.filter((m) => res[m]).map((m, i) => [T(NAMES[m], { bold: true }), ...cols.map(([, k, d]) => T(res[m][k].toFixed(d), { align: "center" }))]
          .map((c) => ({ text: c.text, options: { ...c.options, color: m === best ? HEX.accent1 : HEX.dk1, bold: m === best || c.options.bold, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } })))],
      { x: 8.55, y: 2.6, w: W - M - 8.55, colW: [1.9, 0.75, 0.75, 0.78], fontSize: 13, fontFace: THEME.bodyFontFace, valign: "middle", border: { type: "none" }, margin: [3, 6, 3, 6], rowH: 0.42 },
    );
    const sigTxt = RES.banner ? "" : WINS.length
      ? `${WINS.map((m) => NAMES[m]).join(", ")} beat${WINS.length === 1 ? "s" : ""} both FNN baselines (Diebold–Mariano, p < 0.05). `
      : "No recurrent model beats both FNN baselines significantly. ";
    s.addText(sigTxt + "RMSE, MAE in ft³/min; test = final 12 months, unseen in training and tuning.",
      { x: 8.55, y: 6.0, w: W - M - 8.55, h: 0.6, fontSize: 12, italic: true, color: C.accent4, valign: "top", margin: 0, isTextBox: true });
    s.addNotes(RES.banner
      ? "These are first numbers on the real Muscatine data, but from a deliberately tiny tuning budget, so they only show the pipeline works on real data and give an early direction. The full study (40 sweep runs per model, 10 seeds) is running now and will replace this table. Do not claim a winner from this slide."
      : "Final results: every tuned model had 40 W&B sweep runs and is averaged over 10 seeds. Diebold–Mariano tests compare each recurrent model with both FNN baselines. Be honest about the mixed part: GRU and LSTM are not significantly better than the tuned FNN, and the Window-FNN (same 15-day input, no recurrence) is worse, so the gain comes from recurrence, not from seeing more days.");
  }

  // ===== 6. Conclusion =====
  pres.addSection({ title: "6 Conclusion" });
  {
    const s = content("6 Conclusion", "06 · CONCLUSION", "Conclusion");
    s.addText(
      [
        ...(TOP ? [T(`The ${NAMES[TOP]} beat both lag-FNN baselines on the held-out year: R² ${SUM[TOP].r2.toFixed(2)} vs. ${SUM.fnn_paper.r2.toFixed(2)} (paper FNN) and ${SUM.fnn_tuned.r2.toFixed(2)} (tuned FNN), significant by Diebold–Mariano.`, { bullet: true, breakLine: true, bold: true })] : []),
        T("A complete, tested pipeline compares RNN, GRU and LSTM with the base paper's lag-feature FNN on public full-scale plant data.", { bullet: true, breakLine: true }),
        T("The protocol is fair by design: equal tuning budget, tuned and published baselines, persistence floor, identical test days.", { bullet: true, breakLine: true }),
        T("Leakage is prevented and checked by tests: past-only features, chronological split, train-only scaling.", { bullet: true, breakLine: true }),
        T("Claims rest on 10-seed averages and Diebold–Mariano tests, not on one run.", { bullet: true }),
      ],
      { x: M, y: 1.65, w: 6.6, h: 4.85, fontSize: 17, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 14, isTextBox: true },
    );
    card(s, 7.65, 1.65, W - M - 7.65, 4.85, C.text2, "next");
    s.addText("Next steps", { x: 7.95, y: 1.85, w: 4.5, h: 0.5, fontSize: 20, bold: true, color: C.accent2, margin: 0, isTextBox: true });
    const steps = [RES.banner ? "Finish the full study: 40 runs per model, 10 seeds" : "Explain GRU / LSTM vs. RNN (small data: 467 training days)", "Ablations: raw inputs, strict 1-day-ahead, window 7 / 30", "Write up results in IEEE format", "Future: CH₄ / H₂S targets, multi-step horizons, other plants"];
    for (const [i, t] of steps.entries()) {
      const y = 2.6 + i * 0.95;
      s.addShape(pres.shapes.OVAL, { x: 7.95, y, w: 0.5, h: 0.5, fill: { color: C.accent2 }, line: { type: "none" } });
      s.addText(String(i + 1), { x: 7.95, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(t, { x: 8.65, y: y - 0.12, w: W - M - 8.95, h: 0.75, fontSize: 15, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    }
    s.addNotes("Summarise: the method and the fair protocol are done, tested and already running on real plant data; what remains is the full-budget study, the ablations and the write-up.");
  }

  // ===== 7. References =====
  pres.addSection({ title: "7 References" });
  {
    const s = content("7 References", "07 · REFERENCES", "References");
    const refs = [
      "F. Long, M. Xu, W. Liao, H. Liu, “Reinforcement learning-based modeling of a multiple-year commercial anaerobic digestion system treating manure and food waste,” Energy Convers. Manage., vol. 365, 121811, 2026.",
      "F. Long, M. Xu, W. Liao, H. Liu, “Machine learning for predicting and optimizing the performance of a commercial-scale anaerobic digester with diverse feedstocks and operating conditions,” Bioresour. Technol., 132940, 2025.",
      "D. De Clercq et al., “Interpretable machine learning for predicting biomethane production in industrial-scale anaerobic co-digestion,” Sci. Total Environ., vol. 712, 134574, 2020.",
      "P. Kazemi et al., “Robust data-driven soft sensors for online monitoring of volatile fatty acids in anaerobic digestion processes,” Processes, vol. 8, no. 1, 67, 2020.",
      "R. Gupta et al., “Review of explainable machine learning for anaerobic digestion,” Bioresour. Technol., vol. 369, 128468, 2023.",
      "J.-J. Zhu, M. Yang, Z. J. Ren, “Machine learning in environmental research: Common pitfalls and best practices,” Environ. Sci. Technol., vol. 57, no. 46, pp. 17671–17689, 2023.",
      "J. L. Elman, “Finding structure in time,” Cognitive Science, vol. 14, no. 2, pp. 179–211, 1990.",
      "S. Hochreiter, J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
      "K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proc. EMNLP, 2014.",
      "L. Biewald, “Experiment tracking with Weights and Biases,” 2020. [Online]. Available: wandb.com",
      "F. X. Diebold, R. S. Mariano, “Comparing predictive accuracy,” J. Bus. Econ. Stat., vol. 13, no. 3, pp. 253–263, 1995.",
      "W. K. Newey, K. D. West, “A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix,” Econometrica, vol. 55, no. 3, pp. 703–708, 1987.",
      "H. W. Schroer, C. L. Just, “Feature engineering and supervised machine learning to forecast biogas production during municipal anaerobic co-digestion,” ACS ES&T Eng., vol. 4, no. 3, pp. 660–672, 2024.",
      "H. W. Schroer, C. L. Just, “Laboratory, operational, and SCADA datasets for anaerobic co-digestion at the Muscatine, IA Water Resource Recovery Facility,” Univ. of Iowa, 2024, doi: 10.25820/data.006715.",
    ];
    const half = 7;
    [refs.slice(0, half), refs.slice(half)].forEach((part, c) => {
      s.addText(part.map((r, i) => T(`[${c * half + i + 1}]  ${r}`, { breakLine: i < part.length - 1 })),
        { x: M + c * 6.215, y: 1.65, w: 5.9, h: 5.15, fontSize: 12, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 7, isTextBox: true });
    });
    s.addNotes("IEEE-style references. [1] is the base paper.");
  }

  // ===== 8–11. Additional documents =====
  pres.addSection({ title: "Additional Documents" });
  const docs = [
    ["08 · ADDITIONAL DOCUMENTS", "Phase I completed work — recorded video", "Insert the Phase I video here (Insert → Video → This Device) or paste its link", fa.FaVideo],
    ["09 · ADDITIONAL DOCUMENTS", "Conference attendance certificate", "Pending: to be added after the paper is accepted and presented", fa.FaCertificate],
    ["10 · ADDITIONAL DOCUMENTS", "Conference acceptance letter / email", "Pending: paper is being submitted to a conference; the acceptance email will be added on receipt", fa.FaEnvelopeOpenText],
    ["11 · ADDITIONAL DOCUMENTS", "Conference registration payment proof", "Pending: registration fee will be paid after acceptance", fa.FaReceipt],
  ];
  for (const [tag, title, hint, Ic] of docs) {
    const s = content("Additional Documents", tag, title);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: M, y: 1.65, w: CW, h: 4.95, fill: { color: C.background2 }, line: { color: HEX.accent4, width: 1.25, dashType: "dash" }, rectRadius: 0.12, objectName: "drop zone",
    });
    s.addImage({ data: await icon(Ic, HEX.accent4), x: W / 2 - 0.45, y: 3.0, w: 0.9, h: 0.9 });
    s.addText(hint, { x: M + 1, y: 4.05, w: CW - 2, h: 0.6, fontSize: 16, italic: true, color: C.accent4, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addNotes("Placeholder: replace the dashed box with the document, then delete the box.");
  }

  await pres.writeFile({ fileName: outPath });
  await applyTheme(outPath, THEME);
  console.log("wrote", path.resolve(outPath));
})();
