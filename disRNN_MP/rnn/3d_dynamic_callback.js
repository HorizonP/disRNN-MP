
const plot = document.getElementsByClassName("plotly-graph-div js-plotly-plot")[0];

const colors = plot.data.reduce((map, obj) => {
    map[obj.name] = obj.marker.color;
    return map;
}, { });

const arrs = ['L-', 'L+', 'R-', 'R+'];

// Add 4 line traces initially
const initialTrace = {
    x: [0, 0],
    y: [0, 0],
    z: [0, 0],
    mode: "lines",
    type: "scatter3d",
    hoverinfo: "none",
    showlegend: false,
    visible: false,
};

for (let i = 0; i < 4; i++) {
    Plotly.addTraces(plot, {...initialTrace, name: `line_${i}`, line: {
        width: 5,
        color: colors[arrs[i]]
    }});
}

// Find indices of the line traces
const lineTraceIndices = plot.data.reduce((indices, trace, i) => {
    if (trace.name.startsWith('line_')) {
        indices.push(i);
    }
    return indices;
}, []);

plot.on('plotly_hover', function(data) {
    const hoverPoint = data.points.filter(pt => ['L-', 'L+', 'R-', 'R+'].includes(pt.data.name))[0];

    if (hoverPoint) {
        let hoverCoord = [hoverPoint.x, hoverPoint.y, hoverPoint.z];
        let customData = hoverPoint.customdata;

        if (!customData || customData.length === 0) {
            return; // Exit if customData is not properly defined
        }

        let updateNeeded = false;
        let xUpdate = [];
        let yUpdate = [];
        let zUpdate = [];

        customData.forEach((endCoord, index) => {
            let lineIndex = lineTraceIndices[index];

            if (lineIndex === undefined || !plot.data[lineIndex]) {
                return; // Continue to next iteration if lineIndex is invalid
            }

            let currentTrace = plot.data[lineIndex];

            // Check if the current trace coordinates match the new coordinates
            if (!currentTrace || 
                currentTrace.x[1] !== endCoord[0] || 
                currentTrace.y[1] !== endCoord[1] || 
                currentTrace.z[1] !== endCoord[2]) {
                updateNeeded = true;
                xUpdate.push([hoverCoord[0], endCoord[0]]);
                yUpdate.push([hoverCoord[1], endCoord[1]]);
                zUpdate.push([hoverCoord[2], endCoord[2]]);
            } else {
                xUpdate.push(currentTrace.x);
                yUpdate.push(currentTrace.y);
                zUpdate.push(currentTrace.z);
            }
        });

        // Update all lines at once if needed
        if (updateNeeded) {
            Plotly.restyle(plot, {
                x: xUpdate,
                y: yUpdate,
                z: zUpdate,
                visible: [true, true, true, true]
            }, lineTraceIndices);
        }
    }
});