import React from 'react';
import { Alert, Button } from 'antd';

/** Keep one tab crash from blanking the whole app. */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', error, info);
  }

  render() {
    if (this.state.error) {
      const label = this.props.label || '面板';
      return (
        <Alert
          type="error"
          showIcon
          message={`${label}渲染失败`}
          description={String(this.state.error?.message || this.state.error)}
          action={
            <Button size="small" onClick={() => this.setState({ error: null })}>
              重试
            </Button>
          }
        />
      );
    }
    return this.props.children;
  }
}
