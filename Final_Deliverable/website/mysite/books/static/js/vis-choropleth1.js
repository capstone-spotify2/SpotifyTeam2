/**
 * Created by ruizhao on 10/28/16.
 */
var width1 = $("#choropleth-area").width(),
    height1 = 600;
var legendHeight = 400;

var svg = d3.select("#choropleth-area")
    .append("svg")
    .attr("width", width1)
    .attr("height", height1)
    .style("margin", "30px auto");

svg.style("background-color", "#E6E6F2");

var projection = d3.geo.mercator()
    .scale(400)
    .translate([220, 320]);

var path = d3.geo.path().projection(projection);

var countryDataByID = {};
var choroplethLegend = {};
var mapTopo = {};
//var selectedValue = "UN_Population";


queue()
    .defer(d3.json, "data/africa.topo.json")
    .defer(d3.csv, "data/global-water-sanitation-2015.csv")
    .defer(d3.json, "data/choropleth-legend.json")
    .await(createVisualization);

function createVisualization(error, mapTopJson, countryDataCSV, choroplethLegendData){
    countryDataCSV.forEach(function(d) {
        if(d.WHO_region == "African"){
            countryCode = d.Code.toString();
            countryDataByID[countryCode] = {};
            countryDataByID[countryCode]["Region"] = d.WHO_region;
            countryDataByID[countryCode]["Country"] = d.Country;
            countryDataByID[countryCode]["Code"] = d.Code.toString();
            if (isNaN(d.Improved_Sanitation_2015)) {
                countryDataByID[countryCode]["Improved_Sanitation_2015"] = 0;
            } else {
                countryDataByID[countryCode]["Improved_Sanitation_2015"] = d.Improved_Sanitation_2015;
            }
            if (isNaN(d.Improved_Water_2015)) {
                countryDataByID[countryCode]["Improved_Water_2015"] = 0;
            } else {
                countryDataByID[countryCode]["Improved_Water_2015"] = d.Improved_Water_2015;
            }
            if (isNaN(d.UN_Population)) {
                countryDataByID[countryCode]["UN_Population"] = 0;
            } else {
                countryDataByID[countryCode]["UN_Population"] = d.UN_Population;
            }

        }
    });

    mapTopo = mapTopJson;
    choroplethLegend = choroplethLegendData;
    updateChoropleth();
}


function updateChoropleth() {
    // Dropdown select box updates

    var selectBox = document.getElementById("aspect-type");
    var selectedValue = selectBox.options[selectBox.selectedIndex].value;
    console.log(selectedValue);

    var color_types = choroplethLegend;
    var color = d3.scale.threshold()
        .domain(color_types[selectedValue].color_domain)
        .range(color_types[selectedValue].color_range);

    //tooltip
    var tip = d3.tip()
        .attr('class', 'd3-tip')
        .html(function(d) {
            if(d.properties.adm0_a3_is in countryDataByID){
                if(countryDataByID[d.properties.adm0_a3_is].hasOwnProperty("Code")){
                    if (selectedValue == "Improved_Water_2015") {
                        return "<strong>" + countryDataByID[d.properties.adm0_a3_is].Country + "</strong><br>Population using improved drink-water sources (%) in 2015: <strong>" + countryDataByID[d.properties.adm0_a3_is][selectedValue] + "</strong>";
                    }
                    if (selectedValue == "Improved_Sanitation_2015") {
                        return "<strong>" + countryDataByID[d.properties.adm0_a3_is].Country + "</strong><br>Population using improved sanitation facilities (%) in 2015: <strong>" + countryDataByID[d.properties.adm0_a3_is][selectedValue] + "</strong>";
                    }
                    if (selectedValue == "UN_Population") {
                        return "<strong>" + countryDataByID[d.properties.adm0_a3_is].Country + "</strong><br>UN_Population in 2015: <strong>" + countryDataByID[d.properties.adm0_a3_is][selectedValue] + "</strong>";
                    }
                }
            }else{
                return d.properties.name_long;
            }
        });
    svg.call(tip);

    //Updates choropleth chart
    svg.append("g")
        .attr("class", "region")
        .selectAll("path")
        .data(topojson.feature(mapTopo, mapTopo.objects.collection).features)
        .enter()
        .append("path")
        .attr("d", path)
        .style("fill", function(d) {
            if(d.properties.adm0_a3_is in countryDataByID){
                return color(countryDataByID[d.properties.adm0_a3_is][selectedValue]);
            }else{
                return "#FFFFFF"
            }
        })
        .style("opacity", 0.8)
        .on("mouseover", function(d){
            tip.show(d)
        })
        .on("mouseout", function() {
            tip.hide()
        });

    //Exit legend text and rect
    svg.selectAll("g.legend").remove();
    // Update Legend
    var legend =  svg.selectAll("g.legend")
        .data(color_types[selectedValue].ext_color_domain)
        .enter().append("g")
        .attr("class", "legend");
    // Height and widths for legend
    var ls_w = 20, ls_h = 20;

    // Adds/Updates Rectangles for legend
    legend.append("rect")
        .attr("x", 10)
        .attr("y", function(d, i){ return legendHeight - (i*ls_h) - 2*ls_h;})
        .attr("width", ls_w)
        .attr("height", ls_h)
        .style("fill", function(d, i) { return color(d); })
        .style("opacity", 0.8);

    // Adds/Updates Text for legend
    legend.append("text")
        .attr("x", 50)
        .attr("y", function(d, i){ return legendHeight - (i*ls_h) - ls_h - 4;})
        .text(function(d, i){ return color_types[selectedValue].legend_labels[i]; });
}