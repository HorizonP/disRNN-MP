const plot = document.getElementsByClassName("plotly-graph-div js-plotly-plot")[0]

const colors = plot.data.reduce((map, obj) => {
    map[obj.name] = obj.marker.color;
    return map;
}, { });

function _arrow(coords, color) {
    return {ax:coords[0][0], ay:coords[0][1], x:coords[1][0], y:coords[1][1],
        arrowcolor:color, arrowhead:2, arrowsize:1, arrowwidth:2,
        xref:"x", yref:"y", axref:"x", ayref:'y',
        text:"", showarrow:true}
}

plot.on('plotly_hover', function(data) {
    const pt = data.points[0];
    if (pt) {
        ds = pt.customdata
        let patch = {
            annotations: [
                _arrow([[pt['x'], pt['y']], [ds[0][0],ds[0][1]]], colors['L-']),
                _arrow([[pt['x'], pt['y']], [ds[1][0],ds[1][1]]], colors['L+']),
                _arrow([[pt['x'], pt['y']], [ds[2][0],ds[2][1]]], colors['R-']),
                _arrow([[pt['x'], pt['y']], [ds[3][0],ds[3][1]]], colors['R+'])
            ]
        }
        Plotly.relayout(plot, patch)
    }
})