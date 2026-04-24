import React from 'react';
import './PeriodSelection.css';

function PeriodSelection({ onSelect, activePeriod, periodSelected }) {
  const periods = [
    { id: 'svi', title: 'Сви периоди', description: 'Претражуј целу базу' },
    { id: 'rani_vek', title: 'Рани средњи век', description: 'Рана историја српског народа' },
    { id: 'srednji_vek', title: 'Средњи век', description: 'Српска средњовековна држава' },
    { id: 'novi_vek', title: 'Нови век', description: 'Модерна српска историја' },
    { id: 'ostalo', title: 'Остало', description: 'Општа питања и теме' }
  ];

  return (
    <div className="period-selection">
      <h2>Изаберите период</h2>
      <p>Изаберите историјски период о којем желите да разговарате:</p>
      <div className="period-grid">
        {periods.map(period => (
          <div
            key={period.id}
            className={[
              'period-card',
              period.id === 'svi' ? 'period-svi' : '',
              periodSelected && (activePeriod === null ? period.id === 'svi' : period.id === activePeriod) ? 'period-active' : ''
            ].filter(Boolean).join(' ')}
            onClick={() => onSelect(period.id)}
          >
            <h3>{period.title}</h3>
            <p>{period.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PeriodSelection;
