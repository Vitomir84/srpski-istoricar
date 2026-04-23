import React from 'react';
import './DocumentsPanel.css';
import DocumentItem from './DocumentItem';

function DocumentsPanel({ 
  documents, 
  selectedDocuments, 
  onDocumentToggle, 
  onSelectAll, 
  onDeselectAll 
}) {
  const totalDocuments = documents.length;
  const selectedCount = selectedDocuments.length;

  return (
    <div className="documents-panel">
      <div className="documents-header">
        <h3>📚 Документи</h3>
      </div>
      
      <div className="documents-controls">
        <button className="control-btn" onClick={onSelectAll}>
          Изабери све
        </button>
        <button className="control-btn" onClick={onDeselectAll}>
          Поништи све
        </button>
      </div>
      
      <div className="documents-list">
        {documents.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#999', padding: '20px' }}>
            Учитавање докумената...
          </p>
        ) : (
          documents.map((doc, index) => (
            <DocumentItem
              key={doc.filename}
              document={doc}
              isSelected={selectedDocuments.includes(doc.filename)}
              onToggle={(isSelected) => onDocumentToggle(doc.filename, isSelected)}
            />
          ))
        )}
      </div>
      
      <div className="selected-count">
        {selectedCount === 0 
          ? 'Изабрано: 0 докумената (користи све)'
          : `Изабрано: ${selectedCount} од ${totalDocuments} докумената`
        }
      </div>
    </div>
  );
}

export default DocumentsPanel;
