

// --> CREATE SVG DRAWING AREA
var margin1 = {top: 40, right: 40, bottom: 60, left: 60};

var width1 = $("#choropleth-area").width() - margin1.left - margin1.right,
    height1 = 600 - margin1.top - margin1.bottom;

var svg = d3.select("#choropleth-area").append("svg")
    .attr("width", width1 + margin1.left + margin1.right)
    .attr("height", height1 + margin1.top + margin1.bottom)
    .append("g")
    .attr("transform", "translate(" + margin1.left + "," + margin1.top + ")");

    svg.append("g")
        .style("background-color", "#3E446B");

var africa = {};
var countryData = {};
var countryById = d3.map();

// Use the Queue.js library to read two files
queue()
  .defer(d3.json, "data/africa.topo.json")
  .defer(d3.csv, "data/global-water-sanitation-2015.csv")
  .await(function(error, mapTopJson, countryDataCSV){
    
      //PROCESS DATA
      countryDataCSV.forEach(function(d){
          d.Improved_Sanitation_2015 = +d.Improved_Sanitation_2015;
          d.Improved_Water_2015 = +d.Improved_Water_2015;
          d.UN_Population = +d.UN_Population;
          countryById.set(d.Code, d);
      });

      console.log(countryById.get("AFG").Improved_Sanitation_2015);
      //console.log(countryById.Improved_Sanitation_2015);

      // Convert TopoJSON to GeoJSON (target object = 'countries')
      africa = topojson.feature(mapTopJson, mapTopJson.objects.collection).features;
      countryData = countryDataCSV;

    updateChoropleth();
  });
    

function updateChoropleth() {

    var value1 = d3.select("#aspect-type").property("value");

  // --> Choropleth implementation
    function getColor(d) {
        var dataRow = countryById.get(d.properties.adm0_a3_is);
        if (dataRow) {
            console.log("row", dataRow);
            return colorScale(dataRow[value1]);
        } else {
            console.log("no dataRow", d);
            return "#ccc";
        }
    }

    // function getText(d) {
    //     var dataRow = countryById.get(d.properties.adm0_a3_is);
    //     if (dataRow) {
    //         console.log(dataRow);
    //         return "<strong>" + dataRow.Country + "</strong><br>Population using improved drink-water sources (%) in 2015: <strong>" + dataRow.Improved_Sanitation_2015 + "</strong>";
    //     } else {
    //         console.log("no dataRow", d);
    //         return "<strong>" + d.properties.name + "</strong><br> No data";
    //     }
    // }

    var projection = d3.geo.mercator()
        .scale(380);

    var path = d3.geo.path()
        .projection(projection);

    var colorScale = d3.scale.linear().range(["#FCE4C2", "#EB8A02"]).interpolate(d3.interpolateLab);

    colorScale.domain(d3.extent(countryData, function(d) {return d[value1];}));

    //tooltip
    tip = d3.tip().attr('class', 'd3-tip').html(function(d){
        var dataRow = countryById.get(d.properties.adm0_a3_is);
        if (dataRow) {
            console.log(dataRow);
            if (value1 == "Improved_Water_2015") {
                return "<strong>" + dataRow.Country + "</strong><br>Population using improved drink-water sources (%) in 2015: <strong>" + dataRow.Improved_Water_2015 + "</strong>";
            }
            if (value1 == "Improved_Sanitation_2015") {
                return "<strong>" + dataRow.Country + "</strong><br>Population using improved sanitation facilities (%) in 2015: <strong>" + dataRow.Improved_Sanitation_2015 + "</strong>";
            }
            if (value1 == "UN_Population") {
                return "<strong>" + dataRow.Country + "</strong><br>UN_Population in 2015: <strong>" + dataRow.UN_Population + "</strong>";
            }
        }else {
            console.log("no dataRow", d);
            return "<strong>" + d.properties.name + "</strong><br> No data";
        }
    });

    svg.call(tip);

    var choropleth = svg.selectAll('path.countries')
                        .data(africa);
    choropleth.enter()
        .append('path')
        .attr('class', 'countries')
        .attr('d', path);

    choropleth.attr('fill', function(d,i) {
            /*console.log(d.properties.name);*/
            return getColor(d);
        })
        .on('mouseover', tip.show)
        .on('mouseout', tip.hide);

    // .call(d3.helper.tooltip(
        //     function(d, i){
        //         return getText(d);
        //     }
        //));

    var linear = colorScale;
    //
    // svg.append("g")
    //     .attr("class", "legendLinear")
    //     .attr("transform", "translate(20,20)");
    //
    // var legendLinear = d3.legend.color()
    //     .shapeWidth(30)
    //     .orient('vertical')
    //     .scale(linear);
    //
    // var legend = svg.select(".legendLinear")
    //     .call(legendLinear);
    //
    // choropleth.exit().remove();
    // legend.exit().remove();

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

