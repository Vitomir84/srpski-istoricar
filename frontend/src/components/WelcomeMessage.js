import React from 'react';
import './WelcomeMessage.css';

function WelcomeMessage() {
  return (
    <div className="welcome-message">
      <h2>Добродошли!</h2>
      <p>Постављај питања о српској историји и добићете одговоре базиране на знању из базе података.</p>
      <p style={{ marginTop: '10px', fontSize: '14px' }}>
        Пример: "Ко је био Стефан Немања?" или "Кажи ми о Косовској бици"
      </p>
    </div>
  );
}

export default WelcomeMessage;
