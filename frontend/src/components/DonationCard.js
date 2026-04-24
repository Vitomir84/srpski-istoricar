import React, { useState } from 'react';
import './DonationCard.css';

const ACCOUNT_NUMBER = '105000040532604232';

function DonationCard() {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(ACCOUNT_NUMBER);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = ACCOUNT_NUMBER;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <div className="donate-tab" onClick={() => setIsOpen(prev => !prev)}>
        <span className="donate-heart">♥</span>
        <span className="donate-label">ДОНИРАЈ</span>
      </div>

      <div className={`donate-card${isOpen ? ' open' : ''}`}>
        <button className="donate-close" onClick={() => setIsOpen(false)}>✕</button>
        <div className="donate-icon">♥</div>
        <div className="donate-title">Подржите пројекат</div>
        <div className="donate-text">
          Овај пројекат је настао из љубави према српској историји.<br />
          Свака донација помаже да база знања расте.
        </div>
        <div className="donate-account-label">Број рачуна:</div>
        <div className="donate-account-row">
          <div className="donate-account-number">{ACCOUNT_NUMBER}</div>
          <button
            className={`donate-copy-btn${copied ? ' copied' : ''}`}
            onClick={handleCopy}
          >
            {copied ? '✓ Копирано!' : 'Копирај'}
          </button>
        </div>
        <div className="donate-author">Аутор: Витомир Јовановић</div>
      </div>
    </>
  );
}

export default DonationCard;
