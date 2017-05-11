// SVG drawing area
var margin = {top: 40, right: 40, bottom: 130, left: 60};

var width = $("#chart-area").width() - margin.left - margin.right,
    height = 700 - margin.top - margin.bottom;

var svg1 = d3.select("#chart-area").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Scales
var x = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1);

var y = d3.scale.linear()
    .range([height, 0]);

//axis
var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left");

var yAxisGroup = svg1.append("g")
    .attr("class", "y-axis axis");

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var xAxisGroup = svg1.append("g")
    .attr("class", "x-axis axis");

var labelGroup = svg1.append("g");

var text1 = svg1.append("text")
    .attr("x", 10)
    .attr("y", 10);

text1.text("millions of people");


// Initialize data
loadData();

// UNICEF_beneficiaries data
var data;

// Load CSV file
function loadData() {
    d3.csv("data/unicef-beneficiaries.csv", function(error, csv) {

        csv.forEach(function(d){
            d[2014] = +d[2014];
            d[2015] = +d[2015];
        });

        // Store csv data in global variable
        data = csv;

        // Draw the visualization for the first time
        updateVisualization();
    });
}

// Render visualization
function updateVisualization() {

    var color = d3.scale.category20c();
    // console.log(color(3));
    // console.log(color(7));
    // console.log(color(11));

   console.log(data);
    var value = d3.select("#ranking-type").property("value");
    //console.log(value);

    if(value == "2014") {

        x.domain(data.map(function (d) {
            return d.UNICEF_beneficiaries;
        }));

        y.domain([0, 40]);

        yAxisGroup
            .transition()
            .duration(750)
            .call(yAxis);

        // text1.transition()
        //     .duration(750)
        //     .text("Stores");

        xAxisGroup
            .attr("transform", "translate(0," + height + ")")
            .transition()
            .duration(750)
            .call(xAxis)
            .selectAll("text")
            .attr("transform", "rotate(60)")
            .style("text-anchor", "start");

        var bars = svg1.selectAll(".bar")
            .data(data);

        bars.enter()
            .append("rect")
            .attr("class", "bar");

        bars.attr("fill", function(d){
                if(d.UNICEF_beneficiaries== "Water (development)" ||
                    d.UNICEF_beneficiaries== "Water (emergency)" ||
                    d.UNICEF_beneficiaries== "Water (total)"){
                    return "#336666";
                }
                if(d.UNICEF_beneficiaries== "Sanitation (development)" ||
                    d.UNICEF_beneficiaries== "Sanitation (emergency)" ||
                    d.UNICEF_beneficiaries== "Sanitation (total)"){
                    return "#4F9D9D";
                }
                else{
                    return "#95CACA";
                }
            });

        bars.transition()
            .duration(750)
            .attr("x", function (d) {
                return x(d.UNICEF_beneficiaries);
            })
            .attr("y", function (d) {
                return y(d[2014]);
            })
            .attr("width", x.rangeBand())
            .attr("height", function (d) {
                return height - y(d[2014]);
            });

        var labels = labelGroup.selectAll('text')
            .data(data, function(d) {
                return d[2014];
            });

        labels.enter()
            .append('text');

        labels.attr("x", function(d) {
            return x(d.UNICEF_beneficiaries)+15;
        })
            .attr("y", function(d) {
                return y(d[2014])-3;
            })
            .transition()
            .duration(750)
            .text(function(d) {
                return d[2014];
            });

    }

    else {
        x.domain(data.map(function (d) {
            return d.UNICEF_beneficiaries;
        }));

        y.domain([0,40]);

        yAxisGroup
            .transition()
            .duration(750)
            .call(yAxis);

        xAxisGroup
            .attr("transform", "translate(0," + height + ")")
            .transition()
            .duration(750)
            .call(xAxis)
            .selectAll("text")
            .attr("transform", "rotate(60)")
            .style("text-anchor", "start");

        // text1.transition()
        //     .duration(750)
        //     .text("Billion USD");

        var bars = svg1.selectAll(".bar")
            .data(data);

        bars.enter()
            .append("rect")
            .attr("class", "bar");

        bars.attr("fill", function(d){
                if(d.UNICEF_beneficiaries== "Water (development)" ||
                    d.UNICEF_beneficiaries== "Water (emergency)" ||
                    d.UNICEF_beneficiaries== "Water (total)"){
                    return "#336666";
                }
                if(d.UNICEF_beneficiaries== "Sanitation (development)" ||
                    d.UNICEF_beneficiaries== "Sanitation (emergency)" ||
                    d.UNICEF_beneficiaries== "Sanitation (total)"){
                    return "#4F9D9D";
                }
                else{
                    return "#95CACA";
                }
            });

        bars.transition()
            .duration(750)
            .attr("x", function (d) {
                return x(d.UNICEF_beneficiaries);
            })
            .attr("y", function (d) {
                return y(d[2015]);
            })
            .attr("width", x.rangeBand())
            .attr("height", function (d) {
                return height - y(d[2015]);
            });

        var labels = labelGroup.selectAll('text')
            .data(data, function(d) {
                return d[2015];
            });

        labels.enter()
            .append('text');

        labels.attr("x", function(d) {
            return x(d.UNICEF_beneficiaries)+15;
        })
            .attr("y", function(d) {
                return y(d[2015])-3;
            })
            .transition()
            .duration(750)
            .text(function(d) {
                return d[2015];
            });

    }
    bars.exit().remove();
    labels.exit().remove();
}


