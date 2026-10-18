import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import EntityPanel from "../components/EntityPanel";
import RiskBadge from "../components/RiskBadge";
import client from "../api/client";

export default function MLDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [testText, setTestText] = useState("");
  const [result, setResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const [selectedCell, setSelectedCell] = useState(null);

  useEffect(() => {
    client
      .get("/ml/evaluation")
      .then(({ data: res }) => {
        setData(res);
      })
      .catch((_err) => {
        // Fallback to /ml/comparison if /ml/evaluation is not yet registered
        client
          .get("/ml/comparison")
          .then(({ data: compRes }) => {
            setData({
              status: compRes.evaluation?.status || "unavailable",
              model_info: compRes.model_info || {
                model_name: "CyberSaathi LSTM",
                architecture: "Embedding → SpatialDropout1D → LSTM → Dense layers → Softmax output",
                input_type: "Cybercrime incident description text",
                max_len: 200,
                num_classes: 14,
                classes: compRes.categories || [],
                is_loaded: true,
                is_active_in_workflow: true,
              },
              evaluation: compRes.evaluation || {
                status: "unavailable",
                reason: "Evaluation unavailable — held-out test dataset required",
              },
            });
            setLoading(false);
          })
          .catch((e) => {
            setError(e.response?.data?.detail || "Model insights data not available.");
            setLoading(false);
          });
      });
  }, []);

  const handleTest = async (e) => {
    if (e) e.preventDefault();
    if (!testText.trim()) return;
    setTesting(true);
    setResult(null);
    try {
      const { data: res } = await client.post("/ml/classify", { text: testText });
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || "Classification failed.");
    } finally {
      setTesting(false);
    }
  };

  const samplePrompts = [
    {
      label: "UPI Financial Fraud",
      text: "I received a phone call from someone claiming to be from SBI. They said my debit card was blocked and asked for my OTP. I shared it and immediately ₹45,000 was debited via UPI.",
    },
    {
      label: "Social Media / Hacking",
      text: "Someone hacked my Instagram and WhatsApp accounts, changed my phone number and 2FA, and is now sending fraudulent messages to my family asking for emergency money.",
    },
    {
      label: "Cryptocurrency Fraud",
      text: "I joined a Telegram crypto investment group promising 200% returns in 24 hours. I transferred 1.5 Lakhs worth of USDT to their wallet address and they blocked me.",
    },
    {
      label: "Cyber Extortion",
      text: "I received an extortion email claiming they recorded webcam video and have my browser history. They demanded 50,000 INR in Bitcoin or they will leak it to my contacts.",
    },
    {
      label: "Ambiguous / Greeting",
      text: "Hello, good morning! I need some urgent help with an issue on my phone.",
    },
  ];

  const modelInfo = data?.model_info || {
    model_name: "CyberSaathi LSTM",
    architecture: "Embedding → SpatialDropout1D → LSTM → Dense layers → Softmax output",
    input_type: "Cybercrime incident description text",
    max_len: 200,
    num_classes: 14,
    classes: [
      "Any Other Cyber Crime",
      "Child Pornography CPChild Sexual Abuse Material CSAM",
      "Cryptocurrency Crime",
      "Cyber Attack/ Dependent Crimes",
      "Cyber Terrorism",
      "Hacking  Damage to computercomputer system etc",
      "Online Cyber Trafficking",
      "Online Financial Fraud",
      "Online Gambling  Betting",
      "Online and Social Media Related Crime",
      "Ransomware",
      "RapeGang Rape RGRSexually Abusive Content",
      "Sexually Explicit Act",
      "Sexually Obscene material",
    ],
    is_loaded: true,
    is_active_in_workflow: true,
  };

  const evaluation = data?.evaluation;
  const isEvalAvailable = evaluation?.status === "available" && evaluation?.metrics;

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#020817", color: "#f1f5f9" }}>
      <Navbar />
      <div style={{ maxWidth: "1120px", margin: "0 auto", padding: "40px 24px" }}>
        
        {/* Page Title & Mission */}
        <div style={{ marginBottom: "32px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "8px" }}>
            <span style={{ fontSize: "1.5rem" }}>🧠</span>
            <h1 style={{
              fontFamily: "var(--font-display, sans-serif)",
              fontSize: "1.85rem",
              fontWeight: 700,
              color: "#f8fafc",
              margin: 0,
            }}>
              Model Insights & Evaluation
            </h1>
          </div>
          <p style={{ fontSize: "0.92rem", color: "#94a3b8", margin: 0, maxWidth: "780px", lineHeight: 1.5 }}>
            CyberSaathi uses a pre-trained Deep Learning LSTM neural network as its primary live text classifier.
            Explore the production architecture, evaluation metrics across 14 NCRP categories, and test live predictions in the interactive sandbox.
          </p>
        </div>

        {error && (
          <div style={{
            borderRadius: "10px",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            background: "rgba(239, 68, 68, 0.1)",
            padding: "16px 20px",
            fontSize: "0.88rem",
            color: "#f87171",
            marginBottom: "24px",
          }}>
            {error}
          </div>
        )}

        {/* ─── SECTION 1: PRIMARY LSTM MODEL OVERVIEW ─── */}
        <div style={{
          borderRadius: "14px",
          border: "1px solid rgba(56, 189, 248, 0.28)",
          background: "linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(15, 23, 42, 0.6) 100%)",
          backdropFilter: "blur(14px)",
          padding: "26px",
          marginBottom: "32px",
          boxShadow: "0 10px 30px rgba(0, 0, 0, 0.35)",
        }}>
          {/* Header Row */}
          <div style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "14px",
            borderBottom: "1px solid rgba(56, 189, 248, 0.15)",
            paddingBottom: "18px",
            marginBottom: "20px",
          }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{
                  display: "inline-block",
                  width: "10px",
                  height: "10px",
                  borderRadius: "50%",
                  backgroundColor: modelInfo.is_loaded ? "#22c55e" : "#ef4444",
                  boxShadow: modelInfo.is_loaded ? "0 0 10px #22c55e" : "none",
                }} />
                <h2 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
                  {modelInfo.model_name}
                </h2>
                <span style={{
                  fontSize: "0.72rem",
                  fontFamily: "var(--font-mono, monospace)",
                  fontWeight: 600,
                  padding: "3px 8px",
                  borderRadius: "6px",
                  background: modelInfo.is_loaded ? "rgba(34, 197, 94, 0.15)" : "rgba(239, 68, 68, 0.15)",
                  color: modelInfo.is_loaded ? "#4ade80" : "#f87171",
                  border: `1px solid ${modelInfo.is_loaded ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)"}`,
                }}>
                  {modelInfo.is_loaded ? "STATUS: LOADED" : "STATUS: UNAVAILABLE"}
                </span>
              </div>
              <p style={{ fontSize: "0.82rem", color: "#94a3b8", margin: "6px 0 0" }}>
                Dedicated deep learning sequence classification engine for automated cybercrime triage.
              </p>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
              <span style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                padding: "5px 12px",
                borderRadius: "999px",
                background: "rgba(56, 189, 248, 0.12)",
                color: "#38bdf8",
                border: "1px solid rgba(56, 189, 248, 0.3)",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
              }}>
                <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#38bdf8" }} />
                ACTIVE IN LIVE COMPLAINT & CHAT WORKFLOW
              </span>
            </div>
          </div>

          {/* Model Pipeline Flow */}
          <div style={{ marginBottom: "22px" }}>
            <p style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "0.72rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "#64748b",
              marginBottom: "10px",
            }}>
              Neural Network Architecture Pipeline
            </p>
            <div style={{
              display: "flex",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "8px",
              background: "rgba(2, 8, 23, 0.65)",
              border: "1px solid rgba(56, 189, 248, 0.15)",
              borderRadius: "10px",
              padding: "14px 18px",
              fontSize: "0.82rem",
            }}>
              <span style={{ color: "#94a3b8", fontWeight: 600 }}>Raw Incident Text</span>
              <span style={{ color: "#38bdf8" }}>→</span>
              <span style={{ background: "rgba(59, 130, 246, 0.15)", color: "#93c5fd", padding: "4px 8px", borderRadius: "6px" }}>
                Tokenization (Vocab: 30k)
              </span>
              <span style={{ color: "#38bdf8" }}>→</span>
              <span style={{ background: "rgba(59, 130, 246, 0.15)", color: "#93c5fd", padding: "4px 8px", borderRadius: "6px" }}>
                Padding (MAX_LEN: {modelInfo.max_len})
              </span>
              <span style={{ color: "#38bdf8" }}>→</span>
              <span style={{ background: "rgba(20, 184, 166, 0.15)", color: "#5eead4", padding: "4px 8px", borderRadius: "6px" }}>
                Embedding (128d)
              </span>
              <span style={{ color: "#38bdf8" }}>→</span>
              <span style={{ background: "rgba(20, 184, 166, 0.15)", color: "#5eead4", padding: "4px 8px", borderRadius: "6px" }}>
                SpatialDropout1D
              </span>
              <span style={{ color: "#38bdf8" }}>→</span>
              <span style={{ background: "rgba(168, 85, 247, 0.15)", color: "#c084fc", padding: "4px 8px", borderRadius: "6px", fontWeight: 700 }}>
                LSTM (64 Units)
              </span>
              <span style={{ color: "#38bdf8" }}>→</span>
              <span style={{ background: "rgba(234, 179, 8, 0.15)", color: "#fde047", padding: "4px 8px", borderRadius: "6px" }}>
                Dense (64, ReLU)
              </span>
              <span style={{ color: "#38bdf8" }}>→</span>
              <span style={{ background: "rgba(236, 72, 153, 0.15)", color: "#f472b6", padding: "4px 8px", borderRadius: "6px", fontWeight: 700 }}>
                Softmax (14 Classes)
              </span>
            </div>
          </div>

          {/* Specifications Grid */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
            gap: "14px",
          }}>
            <div style={{
              background: "rgba(2, 8, 23, 0.5)",
              border: "1px solid rgba(148, 163, 184, 0.12)",
              borderRadius: "8px",
              padding: "12px 14px",
            }}>
              <span style={{ fontSize: "0.72rem", color: "#64748b", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                Input Type
              </span>
              <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "#f8fafc", marginTop: "4px" }}>
                {modelInfo.input_type}
              </div>
            </div>

            <div style={{
              background: "rgba(2, 8, 23, 0.5)",
              border: "1px solid rgba(148, 163, 184, 0.12)",
              borderRadius: "8px",
              padding: "12px 14px",
            }}>
              <span style={{ fontSize: "0.72rem", color: "#64748b", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                Max Sequence Length
              </span>
              <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "#38bdf8", marginTop: "4px", fontFamily: "var(--font-mono, monospace)" }}>
                {modelInfo.max_len} tokens
              </div>
            </div>

            <div style={{
              background: "rgba(2, 8, 23, 0.5)",
              border: "1px solid rgba(148, 163, 184, 0.12)",
              borderRadius: "8px",
              padding: "12px 14px",
            }}>
              <span style={{ fontSize: "0.72rem", color: "#64748b", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                Output Taxonomy
              </span>
              <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "#f8fafc", marginTop: "4px" }}>
                {modelInfo.num_classes} NCRP Categories
              </div>
            </div>

            <div style={{
              background: "rgba(2, 8, 23, 0.5)",
              border: "1px solid rgba(148, 163, 184, 0.12)",
              borderRadius: "8px",
              padding: "12px 14px",
            }}>
              <span style={{ fontSize: "0.72rem", color: "#64748b", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                Confidence Threshold
              </span>
              <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "#facc15", marginTop: "4px", fontFamily: "var(--font-mono, monospace)" }}>
                40% (Uncertainty Guard)
              </div>
            </div>
          </div>
        </div>

        {/* ─── SECTION 2: FORMAL EVALUATION METRICS ─── */}
        <div style={{ marginBottom: "36px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
            <div>
              <h2 style={{
                fontFamily: "var(--font-display, sans-serif)",
                fontSize: "1.35rem",
                fontWeight: 700,
                color: "#f8fafc",
                margin: 0,
              }}>
                LSTM Evaluation Metrics
              </h2>
              <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: "4px 0 0" }}>
                Empirical test-set evaluation using scikit-learn across all 14 crime categories.
              </p>
            </div>

            {isEvalAvailable && (
              <span style={{
                fontSize: "0.75rem",
                fontFamily: "var(--font-mono, monospace)",
                padding: "4px 10px",
                borderRadius: "6px",
                background: "rgba(34, 197, 94, 0.12)",
                color: "#4ade80",
                border: "1px solid rgba(34, 197, 94, 0.3)",
              }}>
                TEST EVALUATION ACTIVE · {evaluation.dataset_size} SAMPLES
              </span>
            )}
          </div>

          {/* CASE A: Evaluation Available */}
          {isEvalAvailable ? (
            <>
              {/* Overall Metric Cards */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "14px",
                marginBottom: "20px",
              }}>
                <div style={{
                  borderRadius: "10px",
                  border: "1px solid rgba(56, 189, 248, 0.25)",
                  background: "rgba(15, 23, 42, 0.7)",
                  padding: "16px",
                }}>
                  <span style={{ fontSize: "0.72rem", color: "#94a3b8", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                    Test Accuracy
                  </span>
                  <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#38bdf8", marginTop: "6px", fontFamily: "var(--font-mono, monospace)" }}>
                    {(evaluation.metrics.accuracy * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{
                  borderRadius: "10px",
                  border: "1px solid rgba(52, 211, 153, 0.25)",
                  background: "rgba(15, 23, 42, 0.7)",
                  padding: "16px",
                }}>
                  <span style={{ fontSize: "0.72rem", color: "#94a3b8", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                    Macro Precision
                  </span>
                  <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#34d399", marginTop: "6px", fontFamily: "var(--font-mono, monospace)" }}>
                    {(evaluation.metrics.macro_precision * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{
                  borderRadius: "10px",
                  border: "1px solid rgba(251, 191, 36, 0.25)",
                  background: "rgba(15, 23, 42, 0.7)",
                  padding: "16px",
                }}>
                  <span style={{ fontSize: "0.72rem", color: "#94a3b8", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                    Macro Recall
                  </span>
                  <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#fbbf24", marginTop: "6px", fontFamily: "var(--font-mono, monospace)" }}>
                    {(evaluation.metrics.macro_recall * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{
                  borderRadius: "10px",
                  border: "1px solid rgba(167, 139, 250, 0.25)",
                  background: "rgba(15, 23, 42, 0.7)",
                  padding: "16px",
                }}>
                  <span style={{ fontSize: "0.72rem", color: "#94a3b8", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                    Macro F1-Score
                  </span>
                  <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#a78bfa", marginTop: "6px", fontFamily: "var(--font-mono, monospace)" }}>
                    {(evaluation.metrics.macro_f1 * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{
                  borderRadius: "10px",
                  border: "1px solid rgba(129, 140, 248, 0.25)",
                  background: "rgba(15, 23, 42, 0.7)",
                  padding: "16px",
                }}>
                  <span style={{ fontSize: "0.72rem", color: "#94a3b8", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)" }}>
                    Weighted F1-Score
                  </span>
                  <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#818cf8", marginTop: "6px", fontFamily: "var(--font-mono, monospace)" }}>
                    {(evaluation.metrics.weighted_f1 * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Dataset Metadata sub-banner */}
              <div style={{
                fontSize: "0.78rem",
                color: "#64748b",
                fontFamily: "var(--font-mono, monospace)",
                marginBottom: "24px",
              }}>
                Dataset: {evaluation.dataset_name || "Held-out test split"} · {evaluation.dataset_size} total test samples · Evaluated on{" "}
                {evaluation.evaluated_at ? new Date(evaluation.evaluated_at).toLocaleString() : "—"}
              </div>

              {/* Per-Class Performance Table */}
              <div style={{
                borderRadius: "12px",
                border: "1px solid rgba(56, 189, 248, 0.2)",
                background: "rgba(15, 23, 42, 0.65)",
                overflow: "hidden",
                marginBottom: "28px",
              }}>
                <div style={{ padding: "16px 20px", borderBottom: "1px solid rgba(56, 189, 248, 0.15)" }}>
                  <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
                    Per-Class Performance (All 14 NCRP Categories)
                  </h3>
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", textAlign: "left", fontSize: "0.85rem", borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ background: "rgba(30, 41, 59, 0.6)", color: "#94a3b8" }}>
                        <th style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", fontSize: "0.72rem", textTransform: "uppercase" }}>Crime Category</th>
                        <th style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", fontSize: "0.72rem", textTransform: "uppercase" }}>Precision</th>
                        <th style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", fontSize: "0.72rem", textTransform: "uppercase" }}>Recall</th>
                        <th style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", fontSize: "0.72rem", textTransform: "uppercase" }}>F1-Score</th>
                        <th style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", fontSize: "0.72rem", textTransform: "uppercase" }}>Test Samples (Support)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evaluation.per_class.map((c) => (
                        <tr key={c.category} style={{ borderTop: "1px solid rgba(56, 189, 248, 0.1)", color: "#f8fafc" }}>
                          <td style={{ padding: "12px 18px", fontWeight: 600 }}>{c.category}</td>
                          <td style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", color: "#34d399" }}>
                            {(c.precision * 100).toFixed(1)}%
                          </td>
                          <td style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", color: "#fbbf24" }}>
                            {(c.recall * 100).toFixed(1)}%
                          </td>
                          <td style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)", color: "#a78bfa" }}>
                            {(c.f1_score * 100).toFixed(1)}%
                          </td>
                          <td style={{ padding: "12px 18px", fontFamily: "var(--font-mono, monospace)" }}>
                            <span style={{
                              padding: "2px 8px",
                              borderRadius: "6px",
                              background: c.is_low_support ? "rgba(234, 179, 8, 0.12)" : "rgba(56, 189, 248, 0.12)",
                              color: c.is_low_support ? "#facc15" : "#38bdf8",
                              border: `1px solid ${c.is_low_support ? "rgba(234, 179, 8, 0.25)" : "rgba(56, 189, 248, 0.25)"}`,
                              fontSize: "0.75rem",
                            }}>
                              {c.support} samples {c.is_low_support && "(Low Support)"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Confusion Matrix Heatmap */}
              {evaluation.confusion_matrix && (
                <div style={{
                  borderRadius: "12px",
                  border: "1px solid rgba(56, 189, 248, 0.2)",
                  background: "rgba(15, 23, 42, 0.65)",
                  padding: "20px",
                  marginBottom: "32px",
                }}>
                  <div style={{ marginBottom: "14px" }}>
                    <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
                      14-Class Confusion Matrix
                    </h3>
                    <p style={{ fontSize: "0.78rem", color: "#94a3b8", margin: "4px 0 0" }}>
                      Rows represent actual ground-truth classes; columns represent predicted classes. Diagonal cells represent correct classifications.
                    </p>
                  </div>

                  {/* Responsive Scrollable Matrix */}
                  <div style={{ overflowX: "auto", paddingBottom: "10px" }}>
                    <table style={{ borderCollapse: "collapse", fontSize: "0.72rem", fontFamily: "var(--font-mono, monospace)" }}>
                      <thead>
                        <tr>
                          <th style={{ padding: "6px 8px", color: "#64748b", textAlign: "left", minWidth: "160px" }}>
                            Actual ↓ \ Predicted →
                          </th>
                          {evaluation.confusion_matrix.labels.map((lbl, idx) => (
                            <th key={idx} title={lbl} style={{
                              padding: "6px 4px",
                              color: "#94a3b8",
                              textAlign: "center",
                              width: "36px",
                              maxWidth: "36px",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}>
                              C{idx + 1}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {evaluation.confusion_matrix.matrix.map((row, rowIdx) => {
                          const actualName = evaluation.confusion_matrix.labels[rowIdx];
                          return (
                            <tr key={rowIdx}>
                              <td style={{
                                padding: "6px 8px",
                                color: "#cbd5e1",
                                fontWeight: 600,
                                whiteSpace: "nowrap",
                                borderTop: "1px solid rgba(56, 189, 248, 0.08)",
                              }}>
                                <span style={{ color: "#64748b", marginRight: "6px" }}>C{rowIdx + 1}:</span>
                                {actualName.length > 28 ? actualName.slice(0, 26) + "…" : actualName}
                              </td>
                              {row.map((val, colIdx) => {
                                const isDiagonal = rowIdx === colIdx;
                                const predName = evaluation.confusion_matrix.labels[colIdx];
                                let bgColor = "rgba(15, 23, 42, 0.4)";
                                let textColor = "#64748b";

                                if (isDiagonal) {
                                  if (val > 0) {
                                    bgColor = "rgba(34, 197, 94, 0.28)";
                                    textColor = "#4ade80";
                                  } else {
                                    bgColor = "rgba(15, 23, 42, 0.8)";
                                    textColor = "#475569";
                                  }
                                } else if (val > 0) {
                                  bgColor = "rgba(239, 68, 68, 0.22)";
                                  textColor = "#f87171";
                                }

                                return (
                                  <td
                                    key={colIdx}
                                    onClick={() => setSelectedCell({ actual: actualName, predicted: predName, count: val })}
                                    style={{
                                      padding: "6px 4px",
                                      textAlign: "center",
                                      background: bgColor,
                                      color: textColor,
                                      cursor: "pointer",
                                      border: isDiagonal ? "1px solid rgba(34, 197, 94, 0.3)" : "1px solid rgba(148, 163, 184, 0.08)",
                                      transition: "transform 0.1s",
                                    }}
                                    title={`Actual: ${actualName}\nPredicted: ${predName}\nCount: ${val}`}
                                  >
                                    {val}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Cell Inspector Tooltip Bar */}
                  <div style={{
                    marginTop: "12px",
                    padding: "10px 14px",
                    borderRadius: "6px",
                    background: "rgba(2, 8, 23, 0.75)",
                    border: "1px solid rgba(56, 189, 248, 0.15)",
                    fontSize: "0.8rem",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: "8px",
                  }}>
                    {selectedCell ? (
                      <div>
                        <span style={{ color: "#64748b" }}>Selected Cell: </span>
                        <strong style={{ color: "#f8fafc" }}>Actual: </strong>
                        <span style={{ color: "#38bdf8" }}>{selectedCell.actual}</span>
                        <span style={{ color: "#64748b", margin: "0 8px" }}>→</span>
                        <strong style={{ color: "#f8fafc" }}>Predicted: </strong>
                        <span style={{ color: "#a78bfa" }}>{selectedCell.predicted}</span>
                        <span style={{ color: "#64748b", margin: "0 8px" }}>|</span>
                        <strong style={{ color: "#f8fafc" }}>Count: </strong>
                        <span style={{ color: "#4ade80", fontWeight: 700 }}>{selectedCell.count} samples</span>
                      </div>
                    ) : (
                      <span style={{ color: "#64748b", fontStyle: "italic" }}>
                        💡 Click or hover on any cell in the matrix above to view actual vs predicted incident details.
                      </span>
                    )}
                    <span style={{ fontSize: "0.72rem", color: "#94a3b8", fontFamily: "var(--font-mono, monospace)" }}>
                      Green: Correct · Red: Misclassified
                    </span>
                  </div>
                </div>
              )}
            </>
          ) : (
            /* CASE B: Evaluation Unavailable Card */
            <div style={{
              borderRadius: "14px",
              border: "1px solid rgba(234, 179, 8, 0.35)",
              background: "linear-gradient(180deg, rgba(234, 179, 8, 0.08) 0%, rgba(15, 23, 42, 0.75) 100%)",
              backdropFilter: "blur(12px)",
              padding: "24px 28px",
              marginBottom: "32px",
            }}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <div style={{
                    width: "36px",
                    height: "36px",
                    borderRadius: "8px",
                    background: "rgba(234, 179, 8, 0.15)",
                    border: "1px solid rgba(234, 179, 8, 0.3)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "1.2rem",
                  }}>
                    ⚠
                  </div>
                  <div>
                    <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "#fde047", margin: 0 }}>
                      Evaluation Unavailable — Held-Out Test Dataset Required
                    </h3>
                    <p style={{ fontSize: "0.82rem", color: "#cbd5e1", margin: "4px 0 0" }}>
                      Scientific ML Integrity Guarantee: Formal evaluation metrics must originate from an independent test split.
                    </p>
                  </div>
                </div>

                <span style={{
                  fontSize: "0.72rem",
                  fontFamily: "var(--font-mono, monospace)",
                  fontWeight: 700,
                  padding: "4px 10px",
                  borderRadius: "999px",
                  background: "rgba(234, 179, 8, 0.15)",
                  color: "#facc15",
                  border: "1px solid rgba(234, 179, 8, 0.35)",
                }}>
                  HELD-OUT TEST SET REQUIRED
                </span>
              </div>

              {/* Detailed Explanation */}
              <div style={{
                marginTop: "16px",
                padding: "16px",
                borderRadius: "10px",
                background: "rgba(2, 8, 23, 0.65)",
                border: "1px solid rgba(148, 163, 184, 0.15)",
                fontSize: "0.84rem",
                color: "#94a3b8",
                lineHeight: 1.6,
              }}>
                <p style={{ margin: "0 0 10px", color: "#e2e8f0" }}>
                  {evaluation?.details || (
                    "The CyberSaathi LSTM classifier was pre-trained on an extensive corpus of ~71,422 cybercrime reports across 14 NCRP categories. To prevent data leakage and maintain strict scientific integrity, test performance must be evaluated exclusively against an independent, held-out test split. In accordance with rigorous machine learning standards, training data and synthetic samples are strictly not substituted for genuine evaluation."
                  )}
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "10px", marginTop: "12px" }}>
                  <div style={{ padding: "10px 12px", borderRadius: "6px", background: "rgba(15, 23, 42, 0.8)", border: "1px solid rgba(56, 189, 248, 0.15)" }}>
                    <strong style={{ color: "#38bdf8", display: "block", fontSize: "0.78rem" }}>Missing Data</strong>
                    <span style={{ fontSize: "0.8rem", color: "#cbd5e1" }}>
                      Held-out test dataset (.csv) with ground-truth labels for all 14 NCRP categories.
                    </span>
                  </div>
                  <div style={{ padding: "10px 12px", borderRadius: "6px", background: "rgba(15, 23, 42, 0.8)", border: "1px solid rgba(52, 211, 153, 0.15)" }}>
                    <strong style={{ color: "#34d399", display: "block", fontSize: "0.78rem" }}>Evaluation Protocol</strong>
                    <span style={{ fontSize: "0.8rem", color: "#cbd5e1" }}>
                      Scikit-learn multi-class metrics: Accuracy, Macro F1, Weighted F1, and 14x14 Confusion Matrix.
                    </span>
                  </div>
                </div>
              </div>

              {/* Supported 14 Output Classes */}
              <div style={{ marginTop: "18px" }}>
                <p style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "0.72rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.08em",
                  color: "#cbd5e1",
                  marginBottom: "8px",
                }}>
                  Supported 14 NCRP Classification Classes:
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {modelInfo.classes.map((c, i) => (
                    <span
                      key={i}
                      style={{
                        fontSize: "0.75rem",
                        padding: "3px 9px",
                        borderRadius: "6px",
                        background: "rgba(15, 23, 42, 0.85)",
                        border: "1px solid rgba(56, 189, 248, 0.18)",
                        color: "#94a3b8",
                      }}
                    >
                      <strong style={{ color: "#38bdf8", marginRight: "4px" }}>{i + 1}.</strong> {c}
                    </span>
                  ))}
                </div>
              </div>

              {/* CLI Execution Guide */}
              <div style={{
                marginTop: "18px",
                padding: "12px 16px",
                borderRadius: "8px",
                background: "rgba(2, 8, 23, 0.85)",
                border: "1px solid rgba(234, 179, 8, 0.25)",
              }}>
                <span style={{ fontSize: "0.75rem", color: "#facc15", fontWeight: 600, display: "block", marginBottom: "6px" }}>
                  ⚙ How to run evaluation when a test dataset is provided:
                </span>
                <code style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "0.8rem",
                  color: "#38bdf8",
                  display: "block",
                  background: "rgba(0, 0, 0, 0.4)",
                  padding: "8px 12px",
                  borderRadius: "6px",
                }}>
                  python -m app.ml.evaluate_lstm --test-data path/to/held_out_test_set.csv
                </code>
                <span style={{ fontSize: "0.72rem", color: "#64748b", marginTop: "6px", display: "block" }}>
                  Expected CSV columns: 'text' (or 'complaint_text') and 'category' (or 'crime_category').
                </span>
              </div>
            </div>
          )}
        </div>

        {/* ─── SECTION 3: INTERACTIVE CLASSIFICATION SANDBOX ─── */}
        <div style={{
          borderRadius: "14px",
          border: "1px solid rgba(56, 189, 248, 0.2)",
          background: "rgba(15, 23, 42, 0.7)",
          backdropFilter: "blur(14px)",
          padding: "26px",
        }}>
          <div style={{ marginBottom: "18px" }}>
            <h2 style={{
              fontFamily: "var(--font-display, sans-serif)",
              fontSize: "1.3rem",
              fontWeight: 700,
              color: "#f8fafc",
              margin: 0,
            }}>
              Try the Classifier
            </h2>
            <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: "4px 0 0" }}>
              Enter any cybercrime incident description to test real-time LSTM inference, confidence estimation, and entity extraction.
            </p>
          </div>

          {/* Quick-test prompts */}
          <div style={{ marginBottom: "14px" }}>
            <span style={{ fontSize: "0.72rem", color: "#64748b", textTransform: "uppercase", fontFamily: "var(--font-mono, monospace)", display: "block", marginBottom: "6px" }}>
              Load Example Incidents:
            </span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {samplePrompts.map((s, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setTestText(s.text);
                    setResult(null);
                  }}
                  style={{
                    fontSize: "0.75rem",
                    padding: "4px 10px",
                    borderRadius: "6px",
                    background: "rgba(59, 130, 246, 0.1)",
                    border: "1px solid rgba(59, 130, 246, 0.25)",
                    color: "#93c5fd",
                    cursor: "pointer",
                    transition: "all 0.15s",
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = "rgba(59, 130, 246, 0.2)";
                    e.target.style.borderColor = "#60a5fa";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = "rgba(59, 130, 246, 0.1)";
                    e.target.style.borderColor = "rgba(59, 130, 246, 0.25)";
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleTest}>
            <textarea
              value={testText}
              onChange={(e) => setTestText(e.target.value)}
              rows={4}
              placeholder="e.g. I received a phone call from someone pretending to be my bank manager who asked for my OTP. ₹35,000 was debited via UPI immediately..."
              id="ml-test-textarea"
              style={{
                width: "100%",
                borderRadius: "8px",
                border: "1px solid rgba(56, 189, 248, 0.25)",
                background: "rgba(2, 8, 23, 0.8)",
                padding: "14px 16px",
                fontSize: "0.9rem",
                color: "#f8fafc",
                outline: "none",
                boxSizing: "border-box",
                resize: "vertical",
                lineHeight: 1.5,
              }}
            />
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "12px", flexWrap: "wrap", gap: "10px" }}>
              <button
                type="submit"
                disabled={testing || !testText.trim()}
                id="ml-classify-btn"
                style={{
                  borderRadius: "8px",
                  background: "#f59e0b",
                  border: "none",
                  padding: "10px 26px",
                  fontSize: "0.9rem",
                  fontWeight: 700,
                  color: "#020817",
                  cursor: testing || !testText.trim() ? "not-allowed" : "pointer",
                  opacity: testing || !testText.trim() ? 0.5 : 1,
                  transition: "background 0.2s",
                }}
              >
                {testing ? "Classifying with LSTM..." : "Classify Incident"}
              </button>

              {testText && (
                <button
                  type="button"
                  onClick={() => {
                    setTestText("");
                    setResult(null);
                  }}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "#94a3b8",
                    fontSize: "0.8rem",
                    cursor: "pointer",
                    textDecoration: "underline",
                  }}
                >
                  Clear input
                </button>
              )}
            </div>
          </form>

          {/* Classification Sandbox Output */}
          {result && (
            <div style={{
              marginTop: "24px",
              borderRadius: "12px",
              border: "1px solid rgba(56, 189, 248, 0.25)",
              background: "rgba(2, 8, 23, 0.75)",
              padding: "20px",
            }}>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "12px" }}>
                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px" }}>
                  <span style={{
                    borderRadius: "999px",
                    border: result.confidence < 0.40 ? "1px solid rgba(234, 179, 8, 0.4)" : "1px solid rgba(56, 189, 248, 0.4)",
                    background: result.confidence < 0.40 ? "rgba(234, 179, 8, 0.12)" : "rgba(56, 189, 248, 0.12)",
                    padding: "6px 14px",
                    fontSize: "0.88rem",
                    color: result.confidence < 0.40 ? "#facc15" : "#38bdf8",
                    fontWeight: 700,
                  }}>
                    {result.primary_model ? `${result.primary_model}: ` : ""}{result.category} · {(result.confidence * 100).toFixed(1)}%
                    {result.confidence < 0.40 && " (Uncertain / <40% Threshold)"}
                  </span>
                  <RiskBadge level={result.risk_level} score={result.risk_score} size="sm" />
                </div>

                <span style={{
                  fontSize: "0.72rem",
                  fontFamily: "var(--font-mono, monospace)",
                  color: "#64748b",
                }}>
                  Method: {result.method || "LSTM"}
                </span>
              </div>

              {result.confidence < 0.40 && (
                <div style={{
                  marginTop: "12px",
                  fontSize: "0.8rem",
                  color: "#facc15",
                  background: "rgba(234, 179, 8, 0.08)",
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "1px solid rgba(234, 179, 8, 0.2)",
                }}>
                  ℹ <strong>Low Confidence Alert:</strong> Confidence is below the 40% threshold. The input text lacks specific incident details; classification is marked uncertain and requires clarification.
                </div>
              )}

              {/* Top Alternatives */}
              {result.lstm_alternatives && result.lstm_alternatives.length > 0 && (
                <div style={{ marginTop: "14px", fontSize: "0.8rem", color: "#94a3b8" }}>
                  <strong style={{ color: "#cbd5e1" }}>Top Candidate Alternatives: </strong>
                  {result.lstm_alternatives.map((alt, idx) => (
                    <span
                      key={idx}
                      style={{
                        display: "inline-block",
                        marginRight: "8px",
                        padding: "2px 8px",
                        borderRadius: "4px",
                        background: "rgba(15, 23, 42, 0.8)",
                        border: "1px solid rgba(148, 163, 184, 0.15)",
                        color: "#93c5fd",
                      }}
                    >
                      {alt.category} ({(alt.probability ? alt.probability * 100 : alt.confidence ? alt.confidence * 100 : 0).toFixed(1)}%)
                    </span>
                  ))}
                </div>
              )}

              {/* Missing information warning */}
              {result.missing_fields && result.missing_fields.length > 0 && (
                <div style={{ marginTop: "14px", fontSize: "0.82rem", color: "#94a3b8" }}>
                  <strong style={{ color: "#f8fafc" }}>Missing Complaint Information: </strong>
                  {result.missing_fields.join(", ")}
                </div>
              )}

              {/* Entity extraction panel */}
              <div style={{ marginTop: "18px" }}>
                <EntityPanel entities={result.entities} />
              </div>

              {result.disclaimer && (
                <p style={{ marginTop: "14px", fontSize: "0.74rem", color: "#64748b", fontStyle: "italic", margin: "14px 0 0" }}>
                  ⚖ {result.disclaimer}
                </p>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
