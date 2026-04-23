import React from 'react';
import './DocumentItem.css';

function DocumentItem({ document, isSelected, onToggle }) {
  const handleClick = () => {
    onToggle(!isSelected);
  };

  const handleCheckboxChange = (e) => {
    e.stopPropagation();
    onToggle(e.target.checked);
  };

  return (
    <div 
      className={`document-item ${isSelected ? 'selected' : ''}`}
      onClick={handleClick}
    >
      <div className="document-item-content">
        <input
          type="checkbox"
          className="document-checkbox"
          checked={isSelected}
          onChange={handleCheckboxChange}
          onClick={(e) => e.stopPropagation()}
        />
        <div className="document-info">
          <div className="document-title">
            {document.citation || document.display_name || document.naslov || document.filename}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentItem;
