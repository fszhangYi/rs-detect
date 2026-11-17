import { Col, InputNumber, Row } from 'antd';
import { IDENTITY } from '../lib/matrix.js';

export default function MatrixGrid({ value, onChange }) {
  const mat = value && value.length === 4 ? value : IDENTITY;
  const setCell = (r, c, num) => {
    const next = mat.map(row => row.slice());
    next[r][c] = typeof num === "number" ? num : 0;
    onChange(next);
  };
  return (
    <div>
      {[0,1,2,3].map(r => (
        <Row gutter={6} key={r} style={{ marginBottom: 6 }}>
          {[0,1,2,3].map(c => (
            <Col span={6} key={c} className="matrix-cell">
              <InputNumber
                value={mat[r][c]}
                step={0.001}
                precision={6}
                style={{ width: "100%" }}
                onChange={(v) => setCell(r, c, v)}
              />
            </Col>
          ))}
        </Row>
      ))}
    </div>
  );
}

