var hw = 8;
var hw2 = 64;
let dy = [0, 1, 0, -1, 1, 1, -1, -1];
let dx = [1, 0, -1, 0, 1, -1, 1, -1];
let grid = [
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1]
];
var player = 0;
var ai_player = -1;
var tl = 50;
var tl_idx = -1;
let tl_names = ['レベル1', 'レベル2', 'レベル3', 'レベル4', 'レベル5', 'レベル6', 'レベル7', 'レベル8'];
let tls = [50, 100, 200, 400, 800, 1600, 3200, 6400]
var div_mcts = 20;
var mcts_progress = 0;
var interval_id = -1;
let record = [];
var step = 0;
var direction = -1;
var isstart = true;
var show_value = true;
var show_graph = true;
let graph_values = [];
var ctx = document.getElementById("graph");
var graph = new Chart(ctx, {
    type: 'line',
    data: {
    labels: [],
    datasets: [
        {
        label: '予想勝率',
        data: [],
        fill: false,
        borderColor: "rgb(0,0,0)",
        backgroundColor: "rgb(0,0,0)"
        }
    ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        title: {
            display: false
        },
        legend: {
            display: false
        },
        scales: {
            yAxes: [{
            ticks: {
                max: 100,
                min: 0,
                stepSize: 25,
                callback: function(value, index, values){
                    return  value
                }
            }
            }]
        },
    }
});

const level_range = document.getElementById('ai_level');
const level_show = document.getElementById('ai_level_label');
const setCurrentValue = (val) => {
  level_show.innerText = val;
}
const rangeOnChange = (e) =>{
  setCurrentValue(e.target.value);
}


function start() {
    document.getElementById('start').disabled = true;
    level_range.disabled = true;
    var show_value_elem = document.getElementById('show_value');
    show_value_elem.disabled = true;
    show_value = show_value_elem.checked;
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = true;
    show_graph = show_graph_elem.checked;
    ai_player = -1;
    let players = document.getElementsByName('ai_player');
    for (var i = 0; i < 2; ++i) {
        players.item(i).disabled = true;
        if (players.item(i).checked) {
            ai_player = players.item(i).value;
        }
    }
    console.log(_init_ai(ai_player, 16, 16));
    tl = tls[level_range.value - 1];
    tl_idx = level_range.value - 1;
    console.log("tl", tl);
    if (ai_player == 0){
        direction = 0;
        move(4, 5);
    } else {
        show(-1, -1);
        if (show_value && ai_player != player) {
            var table = document.getElementById("board");
            for (var y = 0; y < 8; ++y) {
                for (var x = 0; x < 8; ++x) {
                    if (grid[y][x] == 2) {
                        table.rows[y].cells[x].firstChild.innerText = "50";
                    }
                }
            }
        }
    }
}

function show(r, c) {
    var table = document.getElementById("board");
    if (!check_mobility()) {
        player = 1 - player;
        if (!check_mobility()) {
            player = 2;
        }
    }
    for (var y = 0; y < 8; ++y) {
        for (var x = 0; x < 8; ++x) {
            table.rows[y].cells[x].style.backgroundColor = "#249972";
            if (grid[y][x] == 0) {
                table.rows[y].cells[x].firstChild.className ="black_stone";
                table.rows[y].cells[x].setAttribute('onclick', "");
            } else if (grid[y][x] == 1) {
                table.rows[y].cells[x].firstChild.className ="white_stone";
                table.rows[y].cells[x].setAttribute('onclick', "");
            } else if (grid[y][x] == 2) {
                if (r == -1 || inside(r, c)) {
                    if (player == 0) {
                        table.rows[y].cells[x].firstChild.className ="legal_stone_black";
                    } else {
                        table.rows[y].cells[x].firstChild.className ="legal_stone_white";
                    }
                    table.rows[y].cells[x].firstChild.innerText = "";
                    if (player != ai_player) {
                        table.rows[y].cells[x].setAttribute('onclick', "move(this.parentNode.rowIndex, this.cellIndex)");
                    } else {
                        table.rows[y].cells[x].setAttribute('onclick', "");
                    }
                } else {
                    table.rows[y].cells[x].firstChild.className ="empty_stone";
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            } else {
                //table.rows[y].cells[x].innerHTML = '<span class="empty_stone"></span>';
                table.rows[y].cells[x].firstChild.className ="empty_stone";
                table.rows[y].cells[x].setAttribute('onclick', "");
            }
        }
    }
    if (inside(r, c)) {
        table.rows[r].cells[c].style.backgroundColor = "#d14141";
    }
    var black_count = 0, white_count = 0;
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (grid[y][x] == 0)
                ++black_count;
            else if (grid[y][x] == 1)
                ++white_count;
        }
    }
    table = document.getElementById("status");
    table.rows[0].cells[2].firstChild.innerHTML = black_count;
    table.rows[0].cells[4].firstChild.innerHTML = white_count;
    if (player == 0) {
        table.rows[0].cells[0].firstChild.className = "legal_stone";
        table.rows[0].cells[6].firstChild.className = "state_blank";
    } else if (player == 1) {
        table.rows[0].cells[0].firstChild.className = "state_blank";
        table.rows[0].cells[6].firstChild.className = "legal_stone";
    } else {
        table.rows[0].cells[0].firstChild.className = "state_blank";
        table.rows[0].cells[6].firstChild.className = "state_blank";
        end_game();
    }
    if (r >= 0){
        if (player == ai_player) {
            ai();
        } else if (show_value) {
            calc_value();
        }
    }
}

function draw(element){
    if (!element) { return; }
    var n = document.createTextNode(' ');
    var disp = element.style.display;
    element.appendChild(n);
    element.style.display = 'none';
    setTimeout(function(){
        element.style.display = disp;
        n.parentNode.removeChild(n);
    },20);
}

function empty(y, x) {
    return grid[y][x] == -1 || grid[y][x] == 2;
}

function inside(y, x) {
    return 0 <= y && y < hw && 0 <= x && x < hw;
}

function check_mobility() {
    var res = false;
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (!empty(y, x))
                continue;
            grid[y][x] = -1;
            for (var dr = 0; dr < 8; ++dr) {
                var ny = y + dy[dr];
                var nx = x + dx[dr];
                if (!inside(ny, nx))
                    continue;
                if (empty(ny, nx))
                    continue;
                if (grid[ny][nx] == player)
                    continue;
                var flag = false;
                var nny = ny, nnx = nx;
                for (var d = 0; d < hw; ++d) {
                    if (!inside(nny, nnx))
                        break;
                    if (empty(nny, nnx))
                        break;
                    if (grid[nny][nnx] == player) {
                        flag = true;
                        break;
                    }
                    nny += dy[dr];
                    nnx += dx[dr];
                }
                if (flag) {
                    grid[y][x] = 2;
                    res = true;
                    break;
                }
            }
        }
    }
    return res;
}

function mcts_main(progress){
    ++mcts_progress;
    _mcts_main();
    progress.value = 100 * mcts_progress / div_mcts;
    progress.innerText = (100 * mcts_progress / div_mcts) + '%';
    //console.log("progress:", 100 * mcts_progress / div_mcts, "%");
    if (mcts_progress == div_mcts){
        mcts_progress = 0;
        clearInterval(interval_id);
        val = _mcts_end();
        var y = Math.floor(val / 1000.0 / hw);
        var x = Math.floor((val - y * 1000.0 * hw) / 1000.0);
        var win_rate = val - y * 1000.0 * hw - x * 1000.0;
        //console.log(y + " " + x + " " + win_rate);
        move(y, x);
        update_graph(win_rate);
        progress.value = 100;
    }
}

function book_main(progress){
    ++mcts_progress;
    _book_main();
    progress.value = 100 * mcts_progress / div_mcts;
    progress.innerText = (100 * mcts_progress / div_mcts) + '%';
    //console.log("progress:", 100 * mcts_progress / div_mcts, "%");
    if (mcts_progress == div_mcts){
        mcts_progress = 0;
        clearInterval(interval_id);
        progress.value = 100;
        val = _book();
        var y = Math.floor(val / 1000.0 / hw);
        var x = Math.floor((val - y * 1000.0 * hw) / 1000.0);
        var win_rate = val - y * 1000.0 * hw - x * 1000.0;
        move(y, x);
        update_graph(win_rate);
        console.log('ai', ai_player, 'pl', player);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function ai() {
    var tl_div = Math.ceil(tl / div_mcts);
    let res = [
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1
    ];
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if(grid[y][x] == 0)
                res[y * hw + x] = 0;
            else if (grid[y][x] == 1)
                res[y * hw + x] = 1;
            else
                res[y * hw + x] = -1;
        }
    }
    var pointer = _malloc(hw2 * 4);
    var offset = pointer / 4;
    HEAP32.set(res, offset);
    var mode = _start_ai(pointer, tl_div, direction);
    _free(pointer);
    console.log('mode', mode);
    var val = -1.0;
    var progress = document.getElementById("progress");
    progress.value = 0;
    if (mode == 0) {
        mcts_progress = 0;
        interval_id = setInterval(mcts_main, 1, progress);
    } else if (mode == 1){
        await sleep(100);
        val = _complete();
        var y = Math.floor(val / 1000.0 / hw);
        var x = Math.floor((val - y * 1000.0 * hw) / 1000.0);
        var win_rate = val - y * 1000.0 * hw - x * 1000.0;
        move(y, x);
        update_graph(win_rate);
        console.log('ai', ai_player, 'pl', player);
    } else {
        mcts_progress = 0;
        interval_id = setInterval(book_main, 1, progress);
    }
}

function calc_value() {
    let res = new Int32Array([
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1
    ]);
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if(grid[y][x] == 0)
                res[y * hw + x] = 0;
            else if (grid[y][x] == 1)
                res[y * hw + x] = 1;
            else
                res[y * hw + x] = -1;
        }
    }
    var n_byte = res.BYTES_PER_ELEMENT;
    var pointer_value = _malloc((hw2 + 10) * n_byte);
    var pointer = _malloc(hw2 * n_byte);
    HEAP32.set(res, pointer / n_byte);
    _calc_value(pointer, 50, direction, pointer_value);
    _free(pointer);
    var output_array = new Int32Array(HEAP32.buffer, pointer_value, hw2 + 10);
    _free(pointer_value);
    console.log(output_array);
    var table = document.getElementById("board");
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (grid[y][x] == 2) {
                table.rows[y].cells[x].firstChild.innerText = output_array[10 + y * hw + x];
            }
        }
    }
}

function move(y, x) {
    if (isstart) {
        isstart = false;
        if (x == 5)
            direction = 0;
        else if (x == 4)
            direction = 1;
        else if (x == 3)
            direction = 3;
        else
            direction = 2;
    }
    grid[y][x] = player;
    for (var dr = 0; dr < 8; ++dr) {
        var ny = y + dy[dr];
        var nx = x + dx[dr];
        if (!inside(ny, nx))
            continue;
        if (empty(ny, nx))
            continue;
        if (grid[ny][nx] == player)
            continue;
        var flag = false;
        var nny = ny, nnx = nx;
        var plus = 0;
        for (var d = 0; d < hw; ++d) {
            if (!inside(nny, nnx))
                break;
            if (empty(nny, nnx))
                break;
            if (grid[nny][nnx] == player) {
                flag = true;
                break;
            }
            nny += dy[dr];
            nnx += dx[dr];
            ++plus;
        }
        if (flag) {
            for (var d = 0; d < plus; ++d) {
                grid[ny + d * dy[dr]][nx + d * dx[dr]] = player;
            }
        }
    }
    ++record.length;
    record[record.length - 1] = [y, x];
    update_record();
    player = 1 - player;
    show(y, x);
}

function update_record() {
    var record_html = document.getElementById('record');
    var new_coord = String.fromCharCode(65 + record[record.length - 1][1]) + String.fromCharCode(49 + record[record.length - 1][0]);
    record_html.innerHTML += new_coord;
}

function update_graph(s) {
    if (show_graph){
        graph.data.labels.push(record.length);
        graph.data.datasets[0].data.push(s);
        graph.update();
    } else {
        let tmp = [record.length, s];
        graph_values.push(tmp);
    }
}

function end_game() {
    for (var i = 0; i < graph_values.length; ++i){
        graph.data.labels.push(graph_values[i][0]);
        graph.data.datasets[0].data.push(graph_values[i][1]);
    }
    graph.update();
    let stones = [0, 0];
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (0 <= grid[y][x] <= 1) {
                ++stones[grid[y][x]];
            }
        }
    }
    var data_json = {};
    if (stones[ai_player] > stones[1 - ai_player]) {
        data_json["a"] = 'win';
    } else {
        data_json["a"] = 'lose';
    }
    html2canvas(document.getElementById('main'),{
        onrendered: function(canvas){
            var imgData = canvas.toDataURL();
            document.getElementById("game_result").src = imgData;
        }
    });
    var tweet_str = "";
    if (stones[ai_player] < stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = "あなたの勝ち！";
        var dis = stones[1 - ai_player] - stones[ai_player];
        tweet_str = "世界17位のオセロAIのレベル8中「" + tl_names[tl_idx] + "」に" + dis + "石勝ちしました！ :)";
    } else if (stones[ai_player] > stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = "AIの勝ち！";
        var dis = stones[ai_player] - stones[1 - ai_player];
        tweet_str = "世界17位のオセロAIのレベル8中「" + tl_names[tl_idx] + "」に" + dis + "石負けしました… :(";
    } else {
        document.getElementById('result_text').innerHTML = "引き分け！";
        tweet_str = "世界17位のオセロAIのレベル8中「" + tl_names[tl_idx] + "」と引き分けました！ :|";
    }
    var tweet_result = document.getElementById('tweet_result');
    tweet_result.innerHTML = '結果をツイート！<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="' + tweet_str + '" data-url="https://www.egaroucid.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>';
    twttr.widgets.load();
    var popup = document.getElementById('js-popup');
    if(!popup) return;
    popup.classList.add('is-show');
    var blackBg = document.getElementById('js-black-bg');
    tweet_result.classList.add('show');
    var new_game = document.getElementById('new_game');
    new_game.classList.add('show');
    closePopUp(blackBg);
    function closePopUp(elem) {
        if(!elem) return;
        elem.addEventListener('click', function() {
            popup.classList.remove('is-show');
            tweet_result.classList.remove('show');
            new_game.classList.remove('show');
        })
    }
}

window.onload = function init() {
    level_range.addEventListener('input', rangeOnChange);
    setCurrentValue(level_range.value);
    var container = document.getElementById('chart_container');
    ctx.clientWidth = container.clientWidth;
    ctx.clientHeight = container.clientHeight;
    grid[3][3] = 1
    grid[3][4] = 0
    grid[4][3] = 0
    grid[4][4] = 1
    player = 0;
    var coord_top = document.getElementById('coord_top');
    var row = document.createElement('tr');
    for (var x = 0; x < hw; ++x) {
        var cell = document.createElement('td');
        cell.className = "coord_cell";
        var coord = document.createElement('span');
        coord.className = "coord";
        coord.innerHTML = String.fromCharCode(65 + x);
        cell.appendChild(coord);
        row.appendChild(cell);
    }
    coord_top.appendChild(row);
    var coord_left = document.getElementById('coord_left');
    for (var y = 0; y < hw; ++y) {
        var row = document.createElement('tr');
        var cell = document.createElement('td');
        cell.className = "coord_cell";
        var coord = document.createElement('span');
        coord.className = "coord";
        coord.innerHTML = y + 1;
        cell.appendChild(coord);
        row.appendChild(cell);
        coord_left.appendChild(row);
    }
    var coord_right = document.getElementById('coord_right');
    for (var y = 0; y < hw; ++y) {
        var row = document.createElement('tr');
        var cell = document.createElement('td');
        cell.className = "coord_cell";
        var coord = document.createElement('span');
        coord.className = "coord";
        cell.appendChild(coord);
        row.appendChild(cell);
        coord_right.appendChild(row);
    }
    var table = document.getElementById('board');
    for (var y = 0; y < hw; ++y) {
        var row = document.createElement('tr');
        for (var x = 0; x < hw; ++x) {
            var cell = document.createElement('td');
            cell.className = "cell";
            var stone = document.createElement('span');
            stone.className = "empty_stone";
            cell.appendChild(stone);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
    show(-2, -2);
    document.getElementById('start').disabled = false;
}