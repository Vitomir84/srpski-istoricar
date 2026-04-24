import React from 'react';
import './SvetiSavaShowcase.css';

function SvetiSavaShowcase() {
  return (
    <div className="sveti-sava-showcase">
      <div className="sveti-ornament-left">
        ☩<br /><span className="ornament-line"></span>
      </div>
      <div className="sveti-sava-frame-outer">
        <div className="sveti-sava-frame-inner">
          <img
            src="/SvetiSavaMileseva.jpg"
            alt="Свети Сава - Милешева"
            className="sveti-sava-img"
          />
        </div>
        <div className="sveti-sava-caption">
          <span className="sveti-caption-title">Свети Сава</span>
          <span className="sveti-caption-sub">Фреска из Милешеве · XIII век</span>
        </div>
      </div>
      <div className="sveti-ornament-right">
        <span className="ornament-line"></span><br />☩
      </div>
    </div>
  );
}

export default SvetiSavaShowcase;
