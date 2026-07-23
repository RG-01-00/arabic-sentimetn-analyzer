import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  ChevronRight,
  FileSpreadsheet,
  History,
  LayoutDashboard,

  Menu,
  MessageSquareText,
  Settings,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import StarsBackground from "./stars/StarsBackground";
import "./App.css";
import * as XLSX from "xlsx";

 function App() {
  const [activePage, setActivePage] = useState("Analyze");
  const [apiStatus, setApiStatus] = useState("checking");
  const [text, setText] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [wakingUp, setWakingUp] = useState(false);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/health",
        {
          signal: AbortSignal.timeout(15000),
        }
      );

      if (response.ok) {
        setApiStatus("connected");
        return true;
      }

      setApiStatus("offline");
      return false;

    } catch (error) {
      setApiStatus("sleeping");
      return false;
    }
  };

  useEffect(() => {
    checkApiStatus();

    const interval = setInterval(() => {
      checkApiStatus();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];

    if (!file) return;

    const MAX_FILE_SIZE = 10 * 1024 * 1024;

    if (file.size > MAX_FILE_SIZE) {
      setError(
        "File is too large. Maximum allowed size is 10 MB."
      );

      setSelectedFile(null);
      setText("");

      event.target.value = "";

      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);

    try {
      const fileName = file.name.toLowerCase();

      if (fileName.endsWith(".txt")) {
        const fileText = await file.text();

        setText(fileText);

        return;
      }

      if (
        fileName.endsWith(".csv") ||
        fileName.endsWith(".xlsx") ||
        fileName.endsWith(".xls")
      ) {
        const arrayBuffer = await file.arrayBuffer();

        const workbook = XLSX.read(arrayBuffer, {
          type: "array",
        });

        const firstSheetName = workbook.SheetNames[0];

        const worksheet =
          workbook.Sheets[firstSheetName];

        const rows = XLSX.utils.sheet_to_json(
          worksheet,
          {
            header: 1,
            defval: "",
          }
        );

        const extractedText = rows
          .flat()
          .filter((cell) => cell !== "")
          .join("\n");

        setText(extractedText);

        return;
      }

      setError(
        "Unsupported file type. Please upload TXT, CSV, XLSX, or XLS."
      );

      setSelectedFile(null);
      setText("");

    } catch (error) {
      console.error("File reading error:", error);

      setError(
        "Could not read the uploaded file."
      );

      setSelectedFile(null);
      setText("");
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setText("");
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    setWakingUp(true);
    setError(null);
    setResult(null);

    try {
      // ============================================================
      // 1. CHECK API
      // ============================================================

      setApiStatus("checking");

      const healthResponse = await fetch(
        "http://127.0.0.1:8000/health",
        {
          signal: AbortSignal.timeout(30000),
        }
      );

      if (!healthResponse.ok) {
        throw new Error(
          `API health check failed: ${healthResponse.status}`
        );
      }

      setApiStatus("connected");
      setWakingUp(false);

      // ============================================================
      // 2. ANALYZE TEXT
      // ============================================================

      const response = await fetch(
        "http://127.0.0.1:8000/analyze",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: text.trim(),
          }),
          signal: AbortSignal.timeout(180000),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          `Analysis failed with status ${response.status}`
        );
      }

      setResult(data);

    } catch (error) {
      console.error("Fetch error:", error);

      if (
        error.name === "TimeoutError" ||
        error.name === "AbortError"
      ) {
        setError(
          "The analysis request timed out. The backend may be loading the AI model or restarting."
        );
      } else {
        setError(error.message);
      }

    } finally {
      setLoading(false);
      setWakingUp(false);
    }
  };

  const handleClear = () => {
    setText("");
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <StarsBackground 
      starCount={200}
      maxDistance={150}
      className="custom-stars"
    >
      <div className="app-shell">
        {/* Mobile overlay */}
        {mobileMenuOpen && (
          <div
            className="mobile-overlay"
            onClick={() => setMobileMenuOpen(false)}
          />
        )}

        {/* Main area */}
        <main className="main-content">
          {/* Header */}
          <header className="topbar">
            <button
              className="mobile-menu-button"
              onClick={() => setMobileMenuOpen(true)}
            >
              <Menu size={22} />
            </button>

            <div className="breadcrumb">
              <div className="brand">
                <div className="brand-icon">
                  <span style={{ fontSize: '30px', fontWeight: 'bolder', color: '#ffffff' }}>ع</span>
                </div>
                <div>
                  <h1>Arabi</h1>
                  <span>RG Smart Platform</span>
                </div>
              </div>
            </div>

            <div className="topbar-right">
             <div className={`api-status ${apiStatus}`}>
                <span className="status-dot" />

                {apiStatus === "checking" && "Checking API..."}
                {apiStatus === "connected" && "API Connected"}
                {apiStatus === "sleeping" && "Server sleeping"}
                {apiStatus === "offline" && "API Offline"}
              </div>
              <div className="avatar">AI</div>
            </div>
          </header>

          {/* Page */}
          <section className="page-container">
            {activePage === "Analyze" && (
              <>
                {/* Page heading */}
                <div className="page-heading">
                  <div>
                    <div className="eyebrow">
                      <Activity size={15} />
                      ARABIC NLP ENGINE
                    </div>
                    <h2>Understand what Arabic text means</h2>
                    <p>
                      Analyze sentiment, detect negation, and identify potential
                      sarcasm in Arabic text
                    </p>
                  </div>
                                <div className={`engine-badge ${apiStatus}`}>
                <span className="pulse-dot" />

                {apiStatus === "checking" && "Checking engine..."}
                {apiStatus === "connected" && "Engine ready"}
                {apiStatus === "sleeping" && "Waking engine..."}
                {apiStatus === "offline" && "Engine unavailable"}
              </div>
                </div>

                {/* Analyzer */}
                <div className="analyzer-grid">
                  <div className="panel input-panel">
                    <div className="panel-header">
                      <div>
                        <h3>Text Analysis</h3>
                        <p>Enter Arabic text to analyze its emotional tone</p>
                      </div>
                      <MessageSquareText size={21} />
                    </div>

                    {/* MANUAL TEXT INPUT */}
                    <div className="textarea-wrapper">
                      <textarea
                        dir="rtl"
                        value={text}
                        onChange={(e) => {
                          setText(e.target.value);
                          setSelectedFile(null);
                        }}
                        placeholder="اكتب النص العربي هنا للتحليل..."
                      />
                      <div className="character-count">
                        {text.length} characters
                      </div>
                    </div>


                  <div className="input-actions">

  {/* LEFT: FILE UPLOAD */}
  <div className="file-upload-section">
    <input
      type="file"
      id="file-upload"
      accept="text/plain,.txt,.csv,.xlsx,.xls"
      onChange={handleFileChange}
      hidden
    />

    <label
      htmlFor="file-upload"
      className="file-upload-button"
    >
      <Upload size={17} />

      <span>
        {selectedFile
          ? selectedFile.name
          : "Upload"}
      </span>
    </label>

    {selectedFile && (
      <button
        type="button"
        className="remove-file-button"
        onClick={handleRemoveFile}
      >
        <X size={16} />
      </button>
    )}
  </div>

  {/* RIGHT: CLEAR + ANALYZE */}
  <div className="action-buttons">

      <button
      className="clear-button"
      onClick={handleClear}
    >
      Clear
</button>

   <button
  className="analyze-button"
  onClick={handleAnalyze}
  disabled={loading}
>
  <Sparkles size={17} />
  {loading
    ? wakingUp 
      ? "Waking up server..." 
      : "Analyzing..."
    : "Analyze Sentiment"}
</button>

  </div>

</div>

                    <div className="example-section">
                      <span>Try an example</span>
                      <button
                        onClick={() =>
                          setText("هذا التطبيق رائع جدا وسهل الاستخدام، أعجبني كثيرا")
                        }
                      >
                        Positive example
                      </button>
                      <button
                        onClick={() =>
                          setText("الخدمة سيئة للغاية والتجربة كانت مخيبة للآمال")
                        }
                      >
                        Negative example
                      </button>
                    </div>

                    
                  </div>

                  {/* Result panel */}
                  <div className="panel result-panel">
                    
                    <div className="panel-header">
                      <div>
                        <h3>Analysis Result</h3>
                        <p>Insights generated by the NLP engine.</p>
                      </div>
                      <BarChart3 size={21} />
                    </div>
                     {/* Display error if any */}
                    {error && <div className="error-message">{error}</div>}
                    {result ? (
                      <div className="result-content">
                        <div className="result-card">
                          {/* Sentiment with emoji */}
                          <div className="sentiment-display">
                            <span className="sentiment-emoji">
                              {result.sentiment_emoji || "📝"}
                            </span>
                            <span className="sentiment-text">
                              {result.sentiment}
                            </span>
                          </div>
                          
                          {/* Confidence score */}
                          <div className="score-display">
                            <span className="score-label">الثقة (Confidence):</span>
                            <span className="score-value">
                              {(result.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          
                         
                          
                          {/* Warning if any */}
                          {result.warning && (
                            <div className="warning-message">
                              ⚠️ {result.warning}
                            </div>
                          )}
                          
                          {/* Mixed sentiment indicator */}
                          {result.mixed && (
                            <div className="mixed-indicator">
                              🤔 هذا النص يحمل مشاعر متضاربة
                            </div>
                          )}
                          
                          {/* Summary */}
                          <div className="summary-text">
                            {result.summary}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="result-placeholder">
                        <div className="placeholder-icon">
                          <Sparkles size={25} />
                        </div>
                        <h3>Your analysis will appear here</h3>
                        <p>
                          Enter Arabic text on the left and run the analysis to
                          reveal sentiment insights                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Feature cards */}
                <div className="section-title">
                  <span>ANALYSIS CAPABILITIES</span>
                </div>

                <div className="capabilities-grid">
                  <CapabilityCard
                    number="01"
                    title="Sentiment Detection"
                    description="Classify Arabic text as positive, negative, or neutral"
                  />
                  <CapabilityCard
                    number="02"
                    title="Negation Awareness"
                    description="Understand how negation can change the meaning of sentiment"
                  />
                  <CapabilityCard
                    number="03"
                    title="Sarcasm Detection"
                    description="Identify linguistic patterns that may indicate potential sarcasm"
                  />
                </div>
              </>
            )}

            {activePage !== "Analyze" && (
              <div className="coming-soon">
                <Sparkles size={30} />
                <h2>{activePage}</h2>
                <p>This section will be built next</p>
              </div>
            )}
          </section>
        </main>
      </div>
    </StarsBackground>
  );
}

function CapabilityCard({ number, title, description }) {
  return (
    <div className="capability-card">
      <span className="capability-number">{number}</span>
      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default App;