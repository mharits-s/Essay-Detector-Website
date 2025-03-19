document.addEventListener('DOMContentLoaded', () => {
    const humanPercentage = {{ results.statistics.human_percentage | default(0) }};
    const aiPercentage = {{ results.statistics.ai_percentage | default(0) }};
    const dominantCategory = humanPercentage > aiPercentage ? "Manusia" : "AI";

    // Update the DOM with the percentages
    document.getElementById('humanPercentage').textContent = humanPercentage;
    document.getElementById('aiPercentage').textContent = aiPercentage;
    document.getElementById('dominantCategoryText').textContent = dominantCategory;

    // Apex Donut Chart Configuration
    var options = {
        chart: {
            height: 160,
            width: 160,
            type: 'donut',
            zoom: { enabled: false }
        },
        plotOptions: {
            pie: {
                donut: {
                    size: '76%',
                    labels: {
                        show: false,
                        total: {
                            show: true,
                            label: dominantCategory,
                            formatter: () => dominantCategory,
                            offsetY: 100
                        }
                    }
                }
            }
        },
        series: [humanPercentage, aiPercentage],
        labels: ['Manusia', 'AI'],
        colors: ['#9747FF', '#D9D9D9'],
        legend: { show: false },
        dataLabels: { enabled: false },
        stroke: { width: 0 },
        tooltip: {
            enabled: true,
            y: { formatter: value => value + '%' }
        }
    };

    var chart = new ApexCharts(document.querySelector("#hs-doughnut-chart"), options);
    chart.render();
});