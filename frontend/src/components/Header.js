import React from 'react';
import './Header.css';

const periodNames = {
  'svi': 'Сви периоди',
  'rani_vek': 'Рани средњи век',
  'srednji_vek': 'Средњи век',
  'novi_vek': 'Нови век',
  'ostalo': 'Остало'
};

function Header({ selectedPeriod, periodSelected }) {
  return (
    <div className="header">
      <h1>Српски историчар</h1>
      <p className="subtitle">Ваш дигитални водич кроз српску историју</p>
      <p>Постављајте питања о историјским документима, личностима и догађајима из српске историје</p>
      <p className="info">
        База знања се састоји од јавно доступних извора са сајта rastko.rs. 
        Постоји могућност додавања нових књига уз сагласност аутора или уколико не крше ауторска права.
      </p>
      {periodSelected && selectedPeriod && (
        <span className="selected-period-badge">
          Период: {periodNames[selectedPeriod] || periodNames['svi']}
        </span>
      )}
      {periodSelected && !selectedPeriod && (
        <span className="selected-period-badge">
          Период: {periodNames['svi']}
        </span>
      )}
    </div>
  );
}

export default Header;
