import React, { Component } from 'react';
import { Routes, Route } from 'react-router-dom';
import ModelDetailPage from './ModelDetail';

export default class HomePage extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div>
                <Routes>
                    <Route path='frontend/' element={<p>This is the home page</p>} />
                    <Route path='/model' element={<ModelDetailPage />} />
                </Routes>
            </div>
        );
    }
}