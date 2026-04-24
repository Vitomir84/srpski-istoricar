import React from 'react';
import './WelcomeMessage.css';

function WelcomeMessage() {
  return (
    <div className="welcome-message">
      <div className="welcome-icon">Добродошли</div>
      <h2>Добродошли!</h2>
      <p>
        Постављајте питања о српској историји и добићете одговоре базиране на знању из базе података. 
        Истражите богату историју српског народа кроз векове.
      </p>
      <div className="example-queries">
        <div className="example-title">Примери питања:</div>
        <div className="example-items">
          <div className="example-item">"Ко је био Стефан Немања?"</div>
          <div className="example-item">"Кажи ми о Косовској бици"</div>
          <div className="example-item">"Опиши владавину царa Душана"</div>
        </div>
      </div>
      
      <div className="author-info">
        <div className="author-line">
          Аутор: <strong>Витомир Јовановић</strong>
        </div>
        <div className="donation-line">
          Донације жиро рачун: <strong>105000040532604232</strong>
        </div>
      </div>
    </div>
  );
}

export default WelcomeMessage;
