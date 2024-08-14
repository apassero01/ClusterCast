import React from 'react';
import useModelData from './ModelData';

function ModelDetailPage({ groupId, clusterId, modelId }) {
    const { data, loading, error } = useModelData(groupId, clusterId, modelId);

    if (loading) {
        return <p>Loading...</p>;
    }
    if (error) {
        return <p>An error occurred: {error.message}</p>;
    }

    console.log(data);
    return (
        <div>
            <h1>{data.train_set_length}</h1>
        </div>
    );
}

export default ModelDetailPage;