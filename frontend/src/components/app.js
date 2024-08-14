// app.js
import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import HomePage from './HomePage';

function App() {
    return (
        <BrowserRouter>
            <div>
                <HomePage />
            </div>
        </BrowserRouter>
    );
}

export default App;
