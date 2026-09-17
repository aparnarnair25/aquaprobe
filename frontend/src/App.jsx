import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const depths = [
  0, 5, 10, 20, 30, 50, 75, 100,
  125, 150, 200, 300, 500, 700, 1000
];

const temperatures = [
  28.4, 28.1, 27.7, 26.9, 25.8,
  24.1, 22.6, 20.8, 19.5, 18.4,
  16.7, 14.2, 10.8, 8.7, 6.2
];

function App() {
  const [lat, setLat] = useState(12.5);
  const [lon, setLon] = useState(68);
  const [date, setDate] = useState("2026-09-17");

  const [sst, setSst] = useState(28);
  const [sss, setSss] = useState(35);
  const [ssh, setSsh] = useState(0);
  const [currentU, setCurrentU] = useState(0);
  const [currentV, setCurrentV] = useState(0);
  const [windU, setWindU] = useState(2);
  const [windV, setWindV] = useState(1);

  const [status, setStatus] = useState("Ready");

  const reconstruct = () => {
    setStatus("Mock reconstruction complete");
  };

  return (
    <div className="app">

      <header className="header">
        <div>
          <h1>🌊 OceanEmbed</h1>
          <p>
            Satellite Embedding-Based Deep Learning Framework for
            Subsurface Ocean Temperature Reconstruction
          </p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Prototype
        </div>
      </header>

      <main className="container">

        <section className="hero">
          <div>
            <h2>North Indian Ocean</h2>
            <p>
              Reconstruct subsurface temperature from surface
              satellite observations
            </p>
          </div>

          <div className="location-box">
            <strong>Selected Point</strong>
            <span>
              {lat.toFixed(2)}°N, {lon.toFixed(2)}°E
            </span>
          </div>
        </section>

        <section className="control-card">

          <h2>Observation Inputs</h2>

          <div className="input-grid">

            <label>
              Latitude
              <input
                type="number"
                value={lat}
                onChange={(e) => setLat(Number(e.target.value))}
              />
            </label>

            <label>
              Longitude
              <input
                type="number"
                value={lon}
                onChange={(e) => setLon(Number(e.target.value))}
              />
            </label>

            <label>
              Date
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
              />
            </label>

            <label>
              SST (°C)
              <input
                type="number"
                value={sst}
                onChange={(e) => setSst(Number(e.target.value))}
              />
            </label>

            <label>
              SSS
              <input
                type="number"
                value={sss}
                onChange={(e) => setSss(Number(e.target.value))}
              />
            </label>

            <label>
              SSH / SLA
              <input
                type="number"
                value={ssh}
                onChange={(e) => setSsh(Number(e.target.value))}
              />
            </label>

            <label>
              Current U
              <input
                type="number"
                value={currentU}
                onChange={(e) => setCurrentU(Number(e.target.value))}
              />
            </label>

            <label>
              Current V
              <input
                type="number"
                value={currentV}
                onChange={(e) => setCurrentV(Number(e.target.value))}
              />
            </label>

            <label>
              Wind U
              <input
                type="number"
                value={windU}
                onChange={(e) => setWindU(Number(e.target.value))}
              />
            </label>

            <label>
              Wind V
              <input
                type="number"
                value={windV}
                onChange={(e) => setWindV(Number(e.target.value))}
              />
            </label>

          </div>

          <button
            className="predict-button"
            onClick={reconstruct}
          >
            🌊 Reconstruct Temperature
          </button>

        </section>

        <section className="dashboard-grid">

          <div className="card map-card">

            <h2>🗺️ Ocean Location</h2>

            <div className="fake-map">

              <div className="map-grid"></div>

              <div
                className="map-point"
                style={{
                  left: `${Math.min(
                    Math.max(((lon - 45) / 60) * 100, 5),
                    95
                  )}%`,
                  top: `${Math.min(
                    Math.max(((30 - lat) / 25) * 100, 5),
                    95
                  )}%`,
                }}
              >
                📍
              </div>

              <div className="map-label">
                North Indian Ocean
              </div>

            </div>

            <p className="map-note">
              Spatial domain: 5°N–30°N, 45°E–105°E
            </p>

          </div>

          <div className="card">

            <h2>🌡️ Temperature Profile</h2>

            <div className="chart-container">
  <ResponsiveContainer width="100%" height={400}>
    <LineChart
      data={depths.map((depth, index) => ({
        depth,
        temperature: temperatures[index],
      }))}
      layout="vertical"
      margin={{
        top: 20,
        right: 30,
        left: 20,
        bottom: 20,
      }}
    >
      <CartesianGrid strokeDasharray="3 3" />

      <XAxis
        type="number"
        label={{
          value: "Temperature (°C)",
          position: "insideBottom",
          offset: -10,
        }}
      />

      <YAxis
        type="number"
        dataKey="depth"
        reversed
        label={{
          value: "Depth (m)",
          angle: -90,
          position: "insideLeft",
        }}
      />

      <Tooltip />

      <Line
        type="monotone"
        dataKey="temperature"
        strokeWidth={3}
        dot={{ r: 3 }}
      />
    </LineChart>
  </ResponsiveContainer>
</div>

            <div className="model-status">
              Model status: <strong>{status}</strong>
            </div>

          </div>

        </section>

        <section className="dashboard-grid">

          <div className="card">

            <h2>🧠 Feature Sensitivity</h2>

            <div className="sensitivity-list">

              {[
                ["SST", 40],
                ["SSS", 15],
                ["SSH / SLA", 15],
                ["Current U", 10],
                ["Current V", 8],
                ["Wind U", 7],
                ["Wind V", 5],
              ].map(([feature, value]) => (

                <div
                  className="sensitivity-row"
                  key={feature}
                >

                  <span>{feature}</span>

                  <div className="bar-background">
                    <div
                      className="bar"
                      style={{
                        width: `${value}%`,
                      }}
                    ></div>
                  </div>

                  <span>
                    {(value / 100).toFixed(2)}
                  </span>

                </div>

              ))}

            </div>

            <p className="small-note">
              Prototype model sensitivity. This represents model
              behavior and not physical causation.
            </p>

          </div>

          <div className="card">

            <h2>📊 Validation</h2>

            <div className="metrics-grid">

              <div>
                <strong>RMSE</strong>
                <span>Pending</span>
              </div>

              <div>
                <strong>MAE</strong>
                <span>Pending</span>
              </div>

              <div>
                <strong>Correlation</strong>
                <span>Pending</span>
              </div>

              <div>
                <strong>Bias</strong>
                <span>Pending</span>
              </div>

            </div>

            <p className="small-note">
              Metrics will be connected to the M3 validation module.
            </p>

          </div>

        </section>

        <section className="card insight-card">

          <h2>💡 Scientific Insight</h2>

          <p>
            Surface observations provide information about the upper
            ocean state. OceanEmbed estimates subsurface temperature
            across 15 standard ocean depths.
          </p>

          <p className="small-note">
            Current outputs are prototype/mock outputs until the
            trained model and validation modules are integrated.
          </p>

        </section>

        <footer>
          <strong>OceanEmbed Prototype</strong>

          <span>
            Hackathon PoC • Synthetic/mock outputs during development
          </span>
        </footer>

      </main>

    </div>
  );
}

export default App;