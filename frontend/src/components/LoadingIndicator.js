import React from 'react';
import './LoadingIndicator.css';

function LoadingIndicator() {
  return (
    <div className="message agent">
      <div className="loading">
        <div className="loading-dot"></div>
        <div className="loading-dot"></div>
        <div className="loading-dot"></div>
      </div>
    </div>
  );
}

export default LoadingIndicator;
