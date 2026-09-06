import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// React 18's createRoot API — enables concurrent features (like streaming updates)
// The older ReactDOM.render() is deprecated in React 18
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
