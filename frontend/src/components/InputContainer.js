import React, { useState } from 'react';
import './InputContainer.css';

function InputContainer({ onSendMessage, disabled, isLoading }) {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    if (message.trim() && !disabled && !isLoading) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="input-wrapper">
      <div className="input-container">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Поставите питање о српској историји..."
          disabled={disabled || isLoading}
          autoComplete="off"
        />
        <button 
          onClick={handleSend} 
          disabled={disabled || isLoading || !message.trim()}
        >
          Пошаљи
        </button>
      </div>
    </div>
  );
}

export default InputContainer;
