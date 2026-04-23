import React from 'react';
import './Message.css';

function Message({ content, isUser }) {
  return (
    <div className={`message ${isUser ? 'user' : 'agent'}`}>
      <div className="message-content">
        {content}
      </div>
    </div>
  );
}

export default Message;
