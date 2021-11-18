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
let bef_grid = [
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1]
];
var n_stones = 4;
var player = 0;
var ai_player = -1;
var depth = 0;
var win_read_depth = 16;
var book_depth = 47;
var level_idx = -1;
let level_names = ['レベル1', 'レベル2', 'レベル3', 'レベル4', 'レベル5', 'レベル6', 'レベル7', 'レベル8', 'レベル9', 'レベル10', 'レベル11', 'レベル12', 'レベル13', 'レベル14', 'カスタム'];
let level_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1];
let level_book = [10, 20, 30, 40, 50, 55, 55, 55, 55, 55, 55, 55, 55, 55, -1];
let level_win_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1]
var game_end = false;
var value_calced = false;
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
        label: '予想最終石差',
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
                suggestedMax: 5.0,
                suggestedMin: -5.0,
                stepSize: 5.0,
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
const custom_setting = document.getElementById('custom');
const book_range = document.getElementById('book');
const read_range = document.getElementById('read');
const win_read_range = document.getElementById('win_read');
const book_label = document.getElementById('book_label');
const read_label = document.getElementById('read_label');
const win_read_label = document.getElementById('win_read_label');
const setCurrentValue = (val) => {
    level_show.innerText = level_names[val];
    if (level_names[val] == 'カスタム'){
        custom_setting.style.display = "block";
    } else {
        custom_setting.style.display = "none";
    }
}

const rangeOnChange = (e) =>{
    setCurrentValue(e.target.value);
}

const setCurrentValue_book = (val) => {
    book_label.innerText = book_label.innerText = book_range.value + '手';
}

const rangeOnChange_book = (e) =>{
    setCurrentValue_book(e.target.value);
}

const setCurrentValue_read = (val) => {
    read_label.innerText = read_label.innerText = read_range.value + '手';
}

const rangeOnChange_read = (e) =>{
    setCurrentValue_read(e.target.value);
}

const setCurrentValue_win_read = (val) => {
    win_read_label.innerText = win_read_label.innerText = win_read_range.value + '手';
}

const rangeOnChange_win_read = (e) =>{
    setCurrentValue_win_read(e.target.value);
}

function start() {
    for (var y = 0; y < hw; ++y){
        for (var x = 0; x < hw; ++x) {
            grid[y][x] = -1;
            bef_grid[y][x] = -1;
        }
    }
    grid[3][3] = 1
    grid[3][4] = 0
    grid[4][3] = 0
    grid[4][4] = 1
    player = 0;
    graph.data.values = [];
    graph.data.labels = [];
    graph.update();
    game_end = false;
    document.getElementById('start').disabled = true;
    level_range.disabled = true;
    var show_value_elem = document.getElementById('show_value');
    show_value_elem.disabled = true;
    show_value = show_value_elem.checked;
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = true;
    book_range.disabled = true;
    read_range.disabled = true;
    win_read_range.disabled = true;
    show_graph = show_graph_elem.checked;
    record = [];
    document.getElementById('record').innerText = '';
    ai_player = -1;
    let players = document.getElementsByName('ai_player');
    for (var i = 0; i < 2; ++i) {
        players.item(i).disabled = true;
        if (players.item(i).checked) {
            ai_player = players.item(i).value;
        }
    }
    depth = level_depth[level_range.value];
    book_depth = level_book[level_range.value];
    win_read_depth = level_win_depth[level_range.value];
    level_idx = level_range.value;
    if (level_names[level_idx] == 'カスタム'){
        depth = read_range.value;
        book_depth = book_range.value;
        win_read_depth = win_read_range.value;
    }
    console.log("depth", depth);
    _init_ai(ai_player, depth, win_read_depth, book_depth);
    console.log("sent params to AI")
    n_stones = 4;
    if (ai_player == 0){
        move(4, 5);
    } else {
        show(-1, -1);
    }
    setInterval(ai_check, 250);
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
                if (bef_grid[y][x] != 0) {
                    table.rows[y].cells[x].innerHTML = '<span class="black_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            } else if (grid[y][x] == 1) {
                if (bef_grid[y][x] != 1) {
                    table.rows[y].cells[x].innerHTML = '<span class="white_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            } else if (grid[y][x] == 2) {
                if (r == -1 || inside(r, c)) {
                    if (player == 0) {
                        table.rows[y].cells[x].firstChild.className ="legal_stone_black";
                    } else {
                        table.rows[y].cells[x].firstChild.className ="legal_stone_white";
                    }
                    table.rows[y].cells[x].setAttribute('onclick', "move(this.parentNode.rowIndex, this.cellIndex)");
                } else {
                    //table.rows[y].cells[x].innerHTML = '<span class="empty_stone"></span>';
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
        game_end = true;
        end_game();
    }
    value_calced = false;
}

function ai_check() {
    if (game_end){
        clearInterval(ai_check);
    } else if (player == ai_player) {
        ai();
    } else if (show_value && !value_calced) {
        calc_value();
        value_calced = true;
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

async function ai() {
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
    var val = _ai(pointer) + 0.005;
    _free(pointer);
    console.log('val', val);
    var y = Math.floor(val / 1000 / hw);
    var x = Math.floor((val - y * 1000 * hw) / 1000);
    var dif_stones = val - y * 1000 * hw - x * 1000 - 100.0;
    console.log('y', y, 'x', x, 'dif_stones', dif_stones);
    move(y, x);
    update_graph(dif_stones);
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
    _calc_value(pointer, 25, direction, pointer_value);
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
    for (var yy = 0; yy < hw; ++yy) {
        for (var xx = 0; xx < hw; ++xx) {
            bef_grid[yy][xx] = grid[yy][xx];
        }
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
    ++n_stones;
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
    html2canvas(document.getElementById('main'),{
        onrendered: function(canvas){
            var imgData = canvas.toDataURL();
            document.getElementById("game_result").src = imgData;
        }
    });
    var tweet_str = "";
    var hint = "ヒントなし";
    if (show_value)
        hint = "ヒントあり"
    if (stones[ai_player] < stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = "あなたの勝ち！";
        var dis = stones[1 - ai_player] - stones[ai_player] + hw2 - stones[ai_player] - stones[1 - ai_player];
        tweet_str = "世界3位のオセロAIの" + level_names[level_idx] + hint + "に" + dis + "石勝ちしました！ :)";
    } else if (stones[ai_player] > stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = "AIの勝ち！";
        var dis = stones[ai_player] - stones[1 - ai_player] + hw2 - stones[ai_player] - stones[1 - ai_player];
        tweet_str = "世界3位のオセロAIの" + level_names[level_idx] + hint + "に" + dis + "石負けしました… :(";
    } else {
        document.getElementById('result_text').innerHTML = "引き分け！";
        tweet_str = "世界3位のオセロAIの" + level_names[level_idx] + hint + "と引き分けました！ :|";
    }
    var tweet_result = document.getElementById('tweet_result');
    tweet_result.innerHTML = '結果をツイート！<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="' + tweet_str + '" data-url="https://www.egaroucid.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>';
    twttr.widgets.load();
    var popup = document.getElementById('js-popup');
    if(!popup) return;
    popup.classList.add('is-show');
    var blackBg = document.getElementById('js-black-bg');
    tweet_result.classList.add('show');
    closePopUp(blackBg);
    function closePopUp(elem) {
        if(!elem) return;
        elem.addEventListener('click', function() {
            popup.classList.remove('is-show');
            tweet_result.classList.remove('show');
            tweet_result.innerHTML = "";
        })
    }
    document.getElementById('start').disabled = false;
    var show_value_elem = document.getElementById('show_value');
    show_value_elem.disabled = false;
    show_value = show_value_elem.checked;
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = false;
    level_range.disabled = false;
    let players = document.getElementsByName('ai_player');
    for (var i = 0; i < 2; ++i)
        players.item(i).disabled = false;
    book_range.disabled = false;
    read_range.disabled = false;
    win_read_range.disabled = false;
}

window.onload = function init() {
    level_range.addEventListener('input', rangeOnChange);
    setCurrentValue(level_range.value);
    book_range.addEventListener('input', rangeOnChange_book);
    setCurrentValue_book(book_range.value);
    read_range.addEventListener('input', rangeOnChange_read);
    setCurrentValue_read(read_range.value);
    win_read_range.addEventListener('input', rangeOnChange_win_read);
    setCurrentValue_win_read(win_read_range.value);
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