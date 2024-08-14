import React, { useState, useEffect } from 'react';

function useModelData(groupId,clusterId, modelId ) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch(`/get_model/${groupId}/${clusterId}/${modelId}`)
            .then(response => response.json())
            .then(data => setData(data))
            .then(() => setLoading(false))
            .catch(setError);
    }, [groupId,clusterId,modelId]);

    return { data, loading, error };
}

export default useModelData;