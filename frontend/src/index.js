import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './components/app';

const container = document.getElementById('app'); // Make sure 'app' is the ID of the root div in your index.html
const root = createRoot(container);

root.render(<App />);
