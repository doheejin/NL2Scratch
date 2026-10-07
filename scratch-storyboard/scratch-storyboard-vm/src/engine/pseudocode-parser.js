/**
 * Creates a block object for Scratch VM.
 * @param {string} opcode - The opcode of the block.
 * @param {object} [inputs={}] - The inputs for the block.
 * @param {object} [fields={}] - The fields for the block.
 * @param {object} [extra={}] - Any extra properties for the block.
 * @returns {object} The block object.
 */
const createBlock = function (id, opcode, inputs = {}, fields = {}, extra = {}) {
    return {
        id,
        opcode,
        next: null,
        parent: null,
        inputs,
        fields,
        ...extra
    };
};

/**
 * Connects an array of block objects by setting their 'next' and 'parent' properties.
 * @param {Array<object>} blocks - The array of block objects to connect.
 * @returns {Array<object>} The connected array of block objects.
 */
const connectBlocks = function (blocks) {
    for (let i = 0; i < blocks.length - 1; i++) {
        blocks[i].next = blocks[i + 1].id;
        blocks[i + 1].parent = blocks[i].id;
    }
    return blocks;
};


/**
 * Generates a random 8-character string ID.
 * @returns {string} The generated ID.
 */
const generateId = function () {
    return Math.random().toString(36)
        .substr(2, 20);
};

const wrapInputBlock = (value, blocks, typeHint, knownVariables, parentId) => {
    const id = generateId();
    let block;

    // const isVariable = knownVariables.find(v => v.name === value); // exact match
    const safeValue = (value === undefined || value === null) ? '' : String(value);
    const isVariable = knownVariables.find(v => v.name.toLowerCase() === safeValue.toLowerCase()); // find closest variable match ? lower case?
    if (isVariable && !(typeHint === 'touching_object_menu' || typeHint === 'sound_menu')) {
        return {name: 'VARIABLE', id: isVariable.id, value: isVariable.name, variableType: isVariable.variableType};
    }

    switch (typeHint) {
    case 'math_number':
        block = createBlock(id, 'math_number', {}, {
            NUM: {name: 'NUM', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'positive_number':
        block = createBlock(id, 'math_positive_number', {}, {
            NUM: {name: 'NUM', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'whole_number':
        block = createBlock(id, 'math_whole_number', {}, {
            NUM: {name: 'NUM', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'angle':
        block = createBlock(id, 'math_angle', {}, {
            NUM: {name: 'NUM', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'text':
        block = createBlock(id, 'text', {}, {
            TEXT: {name: 'TEXT', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'color':
        block = createBlock(id, 'colour_picker', {}, {
            COLOUR: {name: 'COLOUR', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'key_option':
        block = createBlock(id, 'sensing_keyoptions', {}, {
            KEY_OPTION: {name: 'KEY_OPTION', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'goto_menu':
        block = createBlock(id, 'motion_goto_menu', {}, {
            TO: {name: 'TO', id: undefined, value: '_random_'}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'glide_to_menu':
        block = createBlock(id, 'motion_glide_to_menu', {}, {
            TO: {name: 'TO', id: undefined, value: '_random_'}
        }, {id, shadow: true, parent: parentId});
        break;    
    case 'sound_menu':
        block = createBlock(id, 'sound_sounds_menu', {}, {
            SOUND_MENU: {name: 'SOUND_MENU', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'touching_object_menu': {
        let menuValue = value;
        const lower = String(value).toLowerCase().trim();
        if (lower === 'edge' || lower === '_edge_') menuValue = '_edge_';
        if (lower === 'mouse-pointer' || lower === 'mouse pointer' || lower === '_mouse_') menuValue = '_mouse_';
        block = createBlock(id, 'sensing_touchingobjectmenu', {}, {
            TOUCHINGOBJECTMENU: {name: 'TOUCHINGOBJECTMENU', value: menuValue}
        }, {id, shadow: true, parent: parentId});
        break;
    }
    case 'clone_option':
        block = createBlock(id, 'control_create_clone_of_menu', {}, {
            CLONE_OPTION: {name: 'CLONE_OPTION', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    case 'looks_costume':
        block = createBlock(id, 'looks_costume', {}, {
            COSTUME: {name: 'COSTUME', value: value}
        }, {id, shadow: true, parent: parentId});
        break;
    default:
        // Arithmetic Operators: +, -, *, /
        const binaryMathMatch = value.match(/^(.+)\s*([\+\-\*\/])\s*(.+)$/);
        if (binaryMathMatch) {
            const left = binaryMathMatch[1];
            const op = binaryMathMatch[2];
            const right = binaryMathMatch[3];
            let opcode;
            switch (op) {
            case '+': opcode = 'operator_add'; break;
            case '-': opcode = 'operator_subtract'; break;
            case '*': opcode = 'operator_multiply'; break;
            case '/': opcode = 'operator_divide'; break;
            }
            block = createBlock(id, opcode, {
                NUM1: {name: 'NUM1', block: wrapInputBlock(left, blocks, 'math_number', knownVariables, id)},
                NUM2: {name: 'NUM2', block: wrapInputBlock(right, blocks, 'math_number', knownVariables, id)}
            }, {}, {id, parent: parentId});
        }
        // Modulo: `a mod b`
        else if (value.match(/^(.+)\s+mod\s+(.+)$/)) {
            const [, a, b] = value.match(/^(.+)\s+mod\s+(.+)$/);
            block = createBlock(id, 'operator_mod', {
                NUM1: {name: 'NUM1', block: wrapInputBlock(a, blocks, 'math_number', knownVariables, id)},
                NUM2: {name: 'NUM2', block: wrapInputBlock(b, blocks, 'math_number', knownVariables, id)}
            }, {}, {id, parent: parentId});
        }
        // Round: round(x)
        else if (value.match(/^round\s*\((.+)\)$/)) {
            const [, arg] = value.match(/^round\s*\((.+)\)$/);
            block = createBlock(id, 'operator_round', {
                NUM: {name: 'NUM', block: wrapInputBlock(arg, blocks, 'math_number', knownVariables, id)}
            }, {}, {id, parent: parentId});
        }
        // Join: join(x, y)
        else if (value.match(/^join\s*\((.+),\s*(.+)\)$/)) {
            const [, a, b] = value.match(/^join\s*\((.+),\s*(.+)\)$/);
            block = createBlock(id, 'operator_join', {
                STRING1: {name: 'STRING1', block: wrapInputBlock(a, blocks, 'text', knownVariables, id)},
                STRING2: {name: 'STRING2', block: wrapInputBlock(b, blocks, 'text', knownVariables, id)}
            }, {}, {id, parent: parentId});
        }
        // Length of: length of(x)
        else if (value.match(/^length of\s*\((.+)\)$/)) {
            const [, str] = value.match(/^length of\s*\((.+)\)$/);
            block = createBlock(id, 'operator_length', {
                STRING: {name: 'STRING', block: wrapInputBlock(str, blocks, null, knownVariables, id)}
            }, {}, {id, parent: parentId});
        }
        // Pick random: pick random (a) to (b)
        else if (value.match(/^pick random\s*\((.+?)\)\s*to\s*\((.+?)\)$/)) {
            const [, a, b] = value.match(/^pick random\s*\((.+?)\)\s*to\s*\((.+?)\)$/);
            block = createBlock(id, 'operator_pickrandom', {
                FROM: {name: 'FROM', block: wrapInputBlock(a, blocks, 'math_number', knownVariables, id)},
                TO: {name: 'TO', block: wrapInputBlock(b, blocks, 'math_number', knownVariables, id)}
            }, {}, {id, parent: parentId});
        }
        // Letter of: letter n of x
        else if (value.match(/^letter\s+(\d+)\s+of\s+(.+)$/)) {
            const [, letter, str] = value.match(/^letter\s+(\d+)\s+of\s+(.+)$/);
            block = createBlock(id, 'operator_letter_of', {
                LETTER: {name: 'LETTER', block: wrapInputBlock(letter, blocks, 'whole_number', knownVariables, id)},
                STRING: {name: 'STRING', block: wrapInputBlock(str, blocks, null, knownVariables, id)}
            }, {}, {id, parent: parentId});
        }
        // Math functions: sqrt of x, abs of x, etc.
        else if (value.match(/^(abs|floor|ceiling|sqrt|sin|cos|tan|asin|acos|atan|ln|log|e^|10\^)\s+of\s+(.+)$/)) {
            const [, func, arg] = value.match(/^(abs|floor|ceiling|sqrt|sin|cos|tan|asin|acos|atan|ln|log|e\^|10\^)\s+of\s+(.+)$/);
            block = createBlock(id, 'operator_mathop', {
                NUM: {name: 'NUM', block: wrapInputBlock(arg, blocks, 'math_number', knownVariables, id)}
            }, {
                OPERATOR: {name: 'OPERATOR', value: func}
            }, {id, parent: parentId});
        // reporter    
        } else if (value.match(/^x position$/)) {
            block = createBlock(id, 'motion_xposition', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^y position$/)) {
            block = createBlock(id, 'motion_yposition', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^direction$/)) {
            block = createBlock(id, 'motion_direction', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^size$/)) {
            block = createBlock(id, 'looks_size', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^costume \[number v\]$/)) {
            block = createBlock(id, 'looks_costumenumber', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^costume \[name v\]$/)) {
            block = createBlock(id, 'looks_costumename', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^backdrop \[number v\]$/)) {
            block = createBlock(id, 'looks_backdropnumber', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^backdrop \[name v\]$/)) {
            block = createBlock(id, 'looks_backdropname', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^volume$/)) {
            block = createBlock(id, 'sounds_volume', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^answer$/)) {
            block = createBlock(id, 'sensing_answer', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^username$/)) {
            block = createBlock(id, 'sensing_username', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^days since 2000$/)) {
            block = createBlock(id, 'sensing_dayssince2000', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^mouse x$/)) {
            block = createBlock(id, 'sensing_mousex', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^mouse y$/)) {
            block = createBlock(id, 'sensing_mousey', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^loudness$/)) {
            block = createBlock(id, 'sensing_loudness', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^timer$/)) {
            block = createBlock(id, 'sensing_timer', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^current \[year v\]$/)) {
            block = createBlock(id, 'sensing_current_year', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^current \[month v\]$/)) {
            block = createBlock(id, 'sensing_current_month', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^current \[date v\]$/)) {
            block = createBlock(id, 'sensing_current_date', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^current \[day of week v\]$/)) {
            block = createBlock(id, 'sensing_current_day_of_week', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^current \[hour v\]$/)) {
            block = createBlock(id, 'sensing_current_hour', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^current \[minute v\]$/)) {
            block = createBlock(id, 'sensing_current_minute', {}, {}, {id, shadow: true, parent: parentId});
        } else if (value.match(/^current \[second v\]$/)) {
            block = createBlock(id, 'sensing_current_second', {}, {}, {id, shadow: true, parent: parentId}); 
        // default to text block
        } else {
            block = createBlock(id, 'text', {}, {
                TEXT: {name: 'TEXT', value: value}
            }, {id, shadow: true, parent: parentId});
        }
        break;
    } 

    blocks[id] = block;

    return id;
};

const isPureNumber = val => /^-?\d+(\.\d+)?$/.test(val.trim());

/**
 * Parses a condition string and returns an object representing the parsed condition block.
 * @param {string} line - The condition string to parse.
 * @returns {object} Object representing the parsed condition block.
 */
const parseCondition = function (line, blocks, knownVariables, parentId) {
    line = line.trim();
    const id = generateId();
    let block;

    // Logical operators
    if (line.includes(' or ')) {
        const [left, right] = line.split(' or ');
        block = createBlock(id, 'operator_or', {
            OPERAND1: {name: 'OPERAND1', block: parseCondition(left, blocks, knownVariables, id)},
            OPERAND2: {name: 'OPERAND2', block: parseCondition(right, blocks, knownVariables, id)}
        });
    }

    if (line.includes(' and ')) {
        const [left, right] = line.split(' and ');
        block = createBlock(id, 'operator_and', {
            OPERAND1: {name: 'OPERAND1', block: parseCondition(left, blocks, knownVariables, id)},
            OPERAND2: {name: 'OPERAND2', block: parseCondition(right, blocks, knownVariables, id)}
        });
    }

    // Comparisons
    if (line.includes(' = ')) {
        const match = line.match(/\((.+)\) = \((.+)\)/);
        if (match) {
            const left = match[1];
            const right = match[2];
            block = createBlock(id, 'operator_equals', {
                OPERAND1: {name: 'OPERAND1', block: wrapInputBlock(left, blocks, isPureNumber(left) ? 'math_number' : '', knownVariables, id)},
                OPERAND2: {name: 'OPERAND2', block: wrapInputBlock(right, blocks, isPureNumber(right) ? 'math_number' : '', knownVariables, id)}
            });
        }

    }

    if (line.includes(' \> ')) {
        const match = line.match(/\((.+)\) \> \((.+)\)/);
        if (match) {
            const left = match[1];
            const right = match[2];
            block = createBlock(id, 'operator_gt', {
                OPERAND1: {name: 'OPERAND1', block: wrapInputBlock(left, blocks, isPureNumber(left) ? 'math_number' : '', knownVariables, id)},
                OPERAND2: {name: 'OPERAND2', block: wrapInputBlock(right, blocks, isPureNumber(right) ? 'math_number' : '', knownVariables, id)}
            });
        }    
    }

    if (line.includes(' \< ')) {
        const match = line.match(/\((.+)\) \< \((.+)\)/);
        if (match) {
            const left = match[1];
            const right = match[2];
            block = createBlock(id, 'operator_lt', {
                OPERAND1: {name: 'OPERAND1', block: wrapInputBlock(left, blocks, isPureNumber(left) ? 'math_number' : '', knownVariables, id)},
                OPERAND2: {name: 'OPERAND2', block: wrapInputBlock(right, blocks, isPureNumber(right) ? 'math_number' : '', knownVariables, id)}
            });
        }
    }

    // Sensing blocks
    if (line.match(/key \((.*) v\) pressed?/)) {
        const key = line.match(/key \((.*) v\) pressed/)[1];
        block = createBlock(id, 'sensing_keypressed', {
            KEY_OPTION: {name: 'KEY_OPTION', block: wrapInputBlock(key, blocks, 'key_option', knownVariables, id)}
        });
    }

    if (line.match(/touching color \((.+)\)\?/)) {
        const target = line.match(/touching color \((.+)\)\?/);
        block = createBlock(id, 'sensing_touchingcolor', {
            COLOR: {name: 'COLOR', block: wrapInputBlock(target[1], blocks, 'color', knownVariables, id)}
        });
    } else if (line.match(/mouse down\?/i)) {
        block = createBlock(id, 'sensing_mousedown', {}, {});
    } else if (line.match(/mouse up\?/i)) {
        const innerId = generateId();
        const inner = createBlock(innerId, 'sensing_mousedown', {}, {});
        blocks[innerId] = inner;
        block = createBlock(id, 'operator_not', {
            OPERAND: {name: 'OPERAND', block: innerId}
        });
    } else if (line.match(/mouse (left|right)\?/i)) {
        // Scratch doesn't distinguish mouse buttons; treat as generic mouse down.
        block = createBlock(id, 'sensing_mousedown', {}, {});
    } else if (line.match(/touching \((.*) v\)\?/)) {
        const target = line.match(/touching \((.*) v\)\?/)[1];
        block = createBlock(id, 'sensing_touchingobject', {
            TOUCHINGOBJECTMENU: {name: 'TOUCHINGOBJECTMENU', block: wrapInputBlock(target, blocks, 'touching_object_menu', knownVariables, id)}
        });
    }

    if (line.startsWith('not ')) {
        const inner = line.slice(4).trim();
        block = createBlock(id, 'operator_not', {
            OPERAND: {name: 'OPERAND', block: parseCondition(inner, blocks, knownVariables, id)}
        });
    }

    if (line.includes(' contains ')) {
        const target = line.match(/\((.+)\) contains \((.+)\)\?/);
        block = createBlock(id, 'operator_contains', {
            TEXT: {name: 'TEXT', block: wrapInputBlock(target[1], blocks, 'text', knownVariables, id)},
            SUBSTRING: {name: 'SUBSTRING', block: wrapInputBlock(target[2], blocks, 'text', knownVariables, id)}
        });
    }

    if (line.includes(' is touching color ')) {
        const target = line.match(/color \((.+)\) is touching color \((.+)\)\?/);
        block = createBlock(id, 'sensing_coloristouchingcolor', {
            COLOR1: {name: 'COLOR1', block: wrapInputBlock(target[1], blocks, 'color', knownVariables, id)},
            COLOR2: {name: 'COLOR2', block: wrapInputBlock(target[2], blocks, 'color', knownVariables, id)}
        });
    }

    if (block) {
        if (parentId) {
            block.parent = parentId;
        }

        blocks[id] = block;
    }
    return id;

};

const blockPatterns = [
    {
        match: /^say \[(.+)\]$/, // say [Hello!]
        opcode: 'looks_say',
        inputs: (match, blocks, knownVariables, parentId) => ({
            MESSAGE: {name: 'MESSAGE', block: wrapInputBlock(match[1], blocks, 'text', knownVariables, parentId)}
        })
    },
    {
        match: /^say \[(.+)\] for \((.+)\) seconds$/, // say [Hello] for (2) seconds
        opcode: 'looks_sayforsecs',
        inputs: (match, blocks, knownVariables, parentId) => ({
            MESSAGE: {name: 'MESSAGE', block: wrapInputBlock(match[1], blocks, 'text', knownVariables, parentId)},
            SECS: {name: 'SECS', block: wrapInputBlock(match[2], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^think \[(.+)\]$/, // think [Hmm...]
        opcode: 'looks_think',
        inputs: (match, blocks, knownVariables, parentId) => ({
            MESSAGE: {name: 'MESSAGE', block: wrapInputBlock(match[1], blocks, 'text', knownVariables, parentId)}
        })
    },
    {
        match: /^think \[(.+)\] for \((.+)\) seconds$/, // think [Hmm] for (2) seconds
        opcode: 'looks_thinkforsecs',
        inputs: (match, blocks, knownVariables, parentId) => ({
            MESSAGE: {name: 'MESSAGE', block: wrapInputBlock(match[1], blocks, 'text', knownVariables, parentId)},
            SECS: {name: 'SECS', block: wrapInputBlock(match[2], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^switch costume to \[(.+?)\]$/,
        opcode: 'looks_switchcostumeto',
        inputs: (match, blocks, knownVariables, parentId) => ({
            COSTUME: {name: 'COSTUME', block: wrapInputBlock(match[1], blocks, 'look_costume', knownVariables, parentId)}
        })
    },
    {
        match: /^change \[(.+?) v\] by \((.+?)\)$/,
        opcode: 'data_changevariableby',
        inputs: (match, blocks, knownVariables, parentId) => ({
            VALUE: {name: 'VALUE', block: wrapInputBlock(match[2], blocks, 'math_number', knownVariables, parentId)}
        }),
        fields: (match, blocks, knownVariables, parentId) => ({
            // {name: 'VARIABLE', id: 'Al[:dHi#gFpqIbHw/e`g', value: 'Score', variableType: ''}
            VARIABLE: wrapInputBlock(match[1], blocks, 'data_variable', knownVariables, parentId)
        })
    },
    {
        match: /^set \[(.+?) v\] to \[(.+?)\]$/,
        opcode: 'data_setvariableto',
        inputs: (match, blocks, knownVariables, parentId) => ({
            VALUE: {name: 'VALUE', block: wrapInputBlock(match[2], blocks, 'text', knownVariables, parentId)}
        }),
        fields: (match, blocks, knownVariables, parentId) => ({
            VARIABLE: wrapInputBlock(match[1], blocks, 'data_variable', knownVariables, parentId)
        })
    },
    {
        match: /^set \[(.+?) v\] to \((.+?)\)$/,
        opcode: 'data_setvariableto',
        inputs: (match, blocks, knownVariables, parentId) => ({
            VALUE: {name: 'VALUE', block: wrapInputBlock(match[2], blocks, 'math_number', knownVariables, parentId)}
        }),
        fields: (match, blocks, knownVariables, parentId) => ({
            VARIABLE: wrapInputBlock(match[1], blocks, 'data_variable', knownVariables, parentId)
        })
    },
    {
        match: /^point in direction \((.+?)\)$/,
        opcode: 'motion_pointindirection',
        inputs: (match, blocks, knownVariables, parentId) => ({
            DIRECTION: {name: 'DIRECTION', block: wrapInputBlock(match[1], blocks, 'math_angle', knownVariables, parentId)}
        }),
        fields: () => ({})
    },
    {
        match: /^set x to \((.+?)\)$/,
        opcode: 'motion_setx',
        inputs: (match, blocks, knownVariables, parentId) => ({
            X: {name: 'X', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^set y to \((.+?)\)$/,
        opcode: 'motion_sety',
        inputs: (match, blocks, knownVariables, parentId) => ({
            Y: {name: 'Y', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^change x by \((.+?)\)$/,
        opcode: 'motion_changexby',
        inputs: (match, blocks, knownVariables, parentId) => ({
            DX: {name: 'DX', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^change y by \((.+?)\)$/,
        opcode: 'motion_changeyby',
        inputs: (match, blocks, knownVariables, parentId) => ({
            DY: {name: 'DY', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^go to \((.+?) v\)$/, // go to (random position v)
        opcode: 'motion_goto',
        inputs: (match, blocks, knownVariables, parentId) => ({
            TO: {name: 'TO', block: wrapInputBlock(match[1], blocks, 'position', knownVariables, parentId)}
        })
    },
    {
        match: /^go to x: \((.+)\) y: \((.+)\)$/,
        opcode: 'motion_gotoxy',
        inputs: (match, blocks, knownVariables, parentId) => ({
            X: {name: 'X', block: wrapInputBlock(match[1], blocks, null, knownVariables, parentId)},
            Y: {name: 'Y', block: wrapInputBlock(match[2], blocks, null, knownVariables, parentId)}
        })
    },
    {
        match: /^move \((.+)\) steps$/,
        opcode: 'motion_movesteps',
        inputs: (match, blocks, knownVariables, parentId) => ({
            STEPS: {name: 'STEPS', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^glide \((.+)\) secs to \((.+) v\)$/,
        opcode: 'motion_glideto',
        inputs: (match, blocks, knownVariables, parentId) => ({
            SECS: {name: 'SECS', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)},
            TARGET: {name: 'TARGET', block: wrapInputBlock(match[2], blocks, 'glide_to_menu', knownVariables, parentId)}
        })
    },
    {
        match: /^glide \((.+)\) secs to x: \((.+)\) y: \((.+)\)$/,
        opcode: 'motion_glidesecstoxy',
        inputs: (match, blocks, knownVariables, parentId) => ({
            SECS: {name: 'SECS', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)},
            X: {name: 'X', block: wrapInputBlock(match[2], blocks, 'data_variable', knownVariables, parentId)},
            Y: {name: 'Y', block: wrapInputBlock(match[3], blocks, 'data_variable', knownVariables, parentId)}
        })
    },
    {
        match: /^turn (left|right) \((.+?)\) degrees$/,
        opcode: match => (match[1] === 'left' ? 'motion_turnleft' : 'motion_turnright'),
        inputs: (match, blocks, knownVariables, parentId) => ({
            DEGREES: {name: 'DEGREES', block: wrapInputBlock(match[2], blocks, 'angle', knownVariables, parentId)}
        }),
        fields: (match, blocks, knownVariables, parentId) => ({
            DIRECTION: {name: 'DIRECTION', block: wrapInputBlock(match[1], blocks, 'math_number', knownVariables, parentId)}
        })
    },
    {
        match: /^broadcast \[(.+?) v\]$/,
        opcode: 'event_broadcast',
        inputs: (match, blocks, knownVariables, parentId) => ({
            BROADCAST_INPUT: {name: 'BROADCAST_INPUT', block: wrapInputBlock(match[1], blocks, 'broadcast_menu', knownVariables, parentId)}
        })
    },
    {
        match: /^broadcast \[(.+?) v\] and wait$/,
        opcode: 'event_broadcastandwait',
        inputs: (match, blocks, knownVariables, parentId) => ({
            BROADCAST_INPUT: {name: 'BROADCAST_INPUT', block: wrapInputBlock(match[1], blocks, 'broadcast_menu', knownVariables, parentId)}
        })
    },
    {
        match: /^create clone of \[(.+?)\]$/,
        opcode: 'control_create_clone_of',
        inputs: (match, blocks, knownVariables, parentId) => ({
            CLONE_OPTION: {name: 'CLONE_OPTION', block: wrapInputBlock(match[1], blocks, 'clone_option', knownVariables, parentId)}
        })
    },
    {
        match: /^delete this clone$/,
        opcode: 'control_delete_this_clone',
        inputs: (match, blocks, knownVariables, parentId) => ({})
    },
    {
        match: /^stop \[(all|this script|other scripts in sprite) v\]$/,
        opcode: 'control_stop',
        inputs: (match, blocks, knownVariables, parentId) => ({
            STOP_OPTION: {name: 'STOP_OPTION', id: undefined, value: match[1]}
        })
    },
    {
        match: /^wait \((.+?)\) seconds$/,
        opcode: 'control_wait',
        inputs: (match, blocks, knownVariables, parentId) => ({
            DURATION: {name: 'DURATION', block: wrapInputBlock(match[1], blocks, 'positive_number', knownVariables, parentId)}
        })
    },
    {
        match: /^wait until <(.+?)>$/,
        opcode: 'control_wait_until',
        inputs: (match, blocks, knownVariables, parentId) => ({
            CONDITION: {name: 'CONDITION', block: parseCondition(match[1], blocks, knownVariables, parentId)}
        })
    },
    {
        match: /^stop all sounds$/,
        opcode: 'sound_stopallsounds',
        inputs: (match, blocks, knownVariables) => ({})
    },
    {
        match: /^play sound \((.+?) v\)$/,
        opcode: 'sound_play',
        inputs: (match, blocks, knownVariables, parentId) => ({
            SOUND_MENU: {name: 'SOUND_MENU', block: wrapInputBlock(match[1], blocks, 'sound_menu', knownVariables, parentId)}
        })
    },
    {
        match: /^play sound \\((.+) v\\) until done/,
        opcode: 'sound_playuntildone',
        inputs: (match, blocks, knownVariables, parentId) => ({
            SOUND_MENU: {name: 'SOUND_MENU', block: wrapInputBlock(match[1], blocks, 'sound_menu', knownVariables, parentId)}
        })
    },
    {
        match: /hide/,
        opcode: 'looks_hide',
        inputs: (match, blocks, knownVariables) => ({})
    },
    {
        match: /show/,
        opcode: 'looks_show',
        inputs: (match, blocks, knownVariables) => ({})
    },
    {
        match: /^ask (.+) and wait/,
        opcode: 'sensing_askandwait',
        inputs: (match, blocks, knownVariables, parentId) => ({
            QUESTION: {name: 'QUESTION', block: wrapInputBlock(match[1], blocks, 'text', knownVariables, parentId)}
        })
    }

];

/**
* Parses pseudo code into a structured format that can be used to create blocks in Scratch.
* @param {string}  code "pseudo code"
* @param {Array<object>}  globalVariables "global variables names and ids"
* @param {Array<object>}  localVariables "local variables names and ids"
* @param {Array<object>}  targets "targets names and ids"
* @returns {Array<object>} Array of script objects representing parsed blocks.
*/
const parsePseudoCode = function (code, globalVariables = [], localVariables = [], targets = [], sounds = []) {
    const fixedVars = ['size', 'x position', 'y position', 'direction', 'costume #', 'costume name', 
        'volume', 'mouse x', 'mouse y', 'backdrop #', 'backdrop name', 'tempo', 'answer', 'username',
        'timer', 'days since 2000', 'current year', 'current month', 'current day of week', 'current date',
        'current hour', 'current minute', 'current second'].map(variable => ({
        name: variable
    })); // fixed variables are handled as hardcoded not with a variable block with id
    const knownVariables = [...globalVariables, ...localVariables, ...targets, ...sounds];
    const lines = code.split('\n').map(line => line.trim())
        .filter(Boolean);
    const stack = [];
    const scripts = [];

    let currentScript = null;
    let matched = true;
    let matched_this = true;

    for (const line of lines) {
        console.log(line);
        if (line === 'end') {
            if (matched && stack.length > 0) {
                const context = stack.pop();

                // If the context has children, wire them as a SUBSTACK
                if (context.children.length > 0) {
                    context.block.inputs.SUBSTACK = {name: 'SUBSTACK', block: context.children[0].id};
                    context.children[0].parent = context.block.id;
                    connectBlocks(context.children);
                }

                // Set the parent field for the closed control block***
                if (stack.length > 0) {
                    context.block.parent = stack[stack.length - 1].id;
                }

            } else {
                console.warn('At least one line did not match before "end"');
                return [];
            }
            if (!matched) {
                console.warn(`Unhandled line: "${line}"`);
            }
        } else if (line.startsWith('when')) {
            if (currentScript) scripts.push(currentScript);
            currentScript = {
                topBlock: generateId(),
                blocks: {}
            };

            const id = currentScript.topBlock;
            let blockType = '';
            let fields = {};
            let inputs = {};
            matched_this = true;

            // Match different kinds of 'when' blocks
            if (line === 'when @greenFlag clicked' || line === 'when green flag clicked' || line === 'when flag clicked') {
                blockType = 'event_whenflagclicked';
            } else if (line === 'when this sprite clicked') {
                blockType = 'event_whenthisspriteclicked';
            } else if (/when backdrop switches to \[(.+?)\]/.test(line)) {
                const match = line.match(/when backdrop switches to \[(.+?)\]/);
                blockType = 'event_whenbackdropswitchesto';
                fields = {BACKDROP: [match[1], null]};
            } else if (/when I receive \[(.+?)\]/.test(line)) {
                const match = line.match(/when I receive \[(.+?)\]/);
                blockType = 'event_whenbroadcastreceived';
                fields = {BROADCAST_OPTION: {name: 'BROADCAST_OPTION',
                    id:
                    wrapInputBlock(match[1], blocks, 'broadcast_option', knownVariables, id),
                    value: match[1]}};
            } else if (/when \((.+?) v\) key pressed/.test(line)) {
                const match = line.match(/when \((.+?) v\) key pressed/);
                blockType = 'event_whenkeypressed';
                fields = {KEY_OPTION: {name: 'KEY_OPTION', id: undefined, value: match[1]}};
            } else if (/when \[(.+?) v\] > \((.+?)\)/.test(line)) {
                const match = line.match(/when \[(.+?) v\] > \((.+?)\)/);
                blockType = 'event_whengreaterthan';
                fields = {WHENGREATERTHANMENU: [match[1], null]};
                inputs = {VALUE: [1, [10, match[2]]]}; // assuming type 10 is number literal
            } else {
                console.warn(`Unrecognized when block: "${line}"`);
                matched_this = false;
            }
            if (matched_this) {
                currentScript.blocks[id] = createBlock(id, blockType, inputs, fields, {
                    shadow: false,
                    topLevel: true,
                    x: 100,
                    y: 150
                });
            }
        } else if (line === 'forever') {
            const id = generateId();
            const block = createBlock(id, 'control_forever', {}, {});
            currentScript.blocks[id] = block;
            stack[stack.length - 1]?.children.push(block);
            stack.push({id, block, children: []});
            matched_this = true;
        } else if (/^repeat\s*\((.+?)\)/.test(line)) {
            const match = line.match(/^repeat\s*\((.+?)\)/);
            const times = match[1];
            const id = generateId();
            const repId = wrapInputBlock(times, currentScript.blocks, 'whole_number', knownVariables, id);
            const block = createBlock(id, 'control_repeat', {TIMES: {name: 'TIMES', block: repId}}, {});
            currentScript.blocks[id] = block;
            stack[stack.length - 1]?.children.push(block);
            stack.push({id, block, children: []});
            matched_this = true;
        } else if (/^repeat until\s*<(.+?)>/.test(line)) {
            const match = line.match(/^repeat until\s*<(.+?)>/);
            const conditionText = match[1];
            const id = generateId();
            const block = createBlock(id, 'control_repeat_until', {
                CONDITION: {name: 'CONDITION', block: parseCondition(conditionText, currentScript.blocks, knownVariables, id)}
            }, {});
            currentScript.blocks[id] = block;
            stack[stack.length - 1]?.children.push(block);
            stack.push({id, block, children: []});
            matched_this = true;
        } else if (line.startsWith('if')) {
            if (line === 'if on edge, bounce') {
                const id = generateId();
                const block = createBlock(id, 'motion_ifonedgebounce', {}, {});
                currentScript.blocks[id] = block;
                stack[stack.length - 1]?.children.push(block);
                stack.push({id, block, children: []});
                matched_this = true;
            }
            if (line.match(/if\s*<(.+?)>\s*then/)) {
                const id = generateId();
                const condition = line.match(/if\s*<(.+?)>\s*then/)[1];
                const block = createBlock(id, 'control_if', {
                    CONDITION: {name: 'CONDITION', block: parseCondition(condition, currentScript.blocks, knownVariables, id)}
                }, {});
                currentScript.blocks[id] = block;
                stack[stack.length - 1]?.children.push(block);
                stack.push({id, block, children: []});
                matched_this = true;
            } else {
                console.warn(`Unrecognized if condition: "${line}"`);
                matched_this = false;
            }
        } else if (line.startsWith('else')) {
            if (stack.length > 0) {
                const context = stack[stack.length - 1];
                if (context.block.opcode === 'control_if') {
                    // Create if-else block
                    const id = generateId();
                    const ifElseBlock = createBlock(id, 'control_if_else', {}, {});

                    // Transfer the CONDITION and SUBSTACK from the original 'if'
                    ifElseBlock.inputs.CONDITION = context.block.inputs.CONDITION;
                    ifElseBlock.inputs.SUBSTACK = context.block.inputs.SUBSTACK;

                    // Replace the old block in the parent context
                    const parentContext = stack[stack.length - 1];
                    parentContext.children = parentContext.children.map(child =>
                        (child === context.block ? ifElseBlock : child)
                    );

                    // Push context to fill SUBSTACK2 (the else body)
                    stack.push({
                        id,
                        block: ifElseBlock,
                        children: []
                    });
                    matched_this = true;
                } else {
                    console.warn('Else without matching if block');
                    matched_this = false;
                }
            } else {
                console.warn('Else without any if block in stack');
                matched_this = false;
            }
        } else if (line.startsWith('go to (random position v)')) {
            const id = generateId();
            currentScript.blocks[id] = createBlock(id, 'motion_goto', {
                TO: {name: 'TO', block: wrapInputBlock('_random_', currentScript.blocks, 'goto_menu', knownVariables, id)}
            }, {});
            stack[stack.length - 1]?.children.push(currentScript.blocks[id]);
            matched_this = true;
        } else {
            for (const pattern of blockPatterns) {
                const match = line.match(pattern.match);
                if (match && currentScript) {
                    const id = generateId();
                    const opcode = typeof pattern.opcode === 'function' ? pattern.opcode(match) : pattern.opcode;
                    const inputs = pattern.inputs ? pattern.inputs(match, currentScript.blocks, knownVariables, id) : {};
                    const fields = pattern.fields ? pattern.fields(match, currentScript.blocks, knownVariables, id) : {};

                    currentScript.blocks[id] = createBlock(id, opcode, inputs, fields, {
                        shadow: false,
                        topLevel: false
                    });
                    stack[stack.length - 1]?.children.push(currentScript.blocks[id]);
                    matched_this = true;
                    break;
                }
                matched_this = false;
            }
        }
        matched = matched && matched_this;
        console.log(matched_this);
    }
    console.log('Matched All: ', matched);

    if (currentScript && matched) {
        // Connect top-level blocks outside any control block
        // Only include blocks not already referenced in any inputs
        const usedBlockIds = new Set();

        // Collect all blocks referenced as inputs (e.g. SUBSTACKs or CONDITIONs)
        for (const block of Object.values(currentScript.blocks)) {
            for (const input of Object.values(block.inputs || {})) {
                if (Array.isArray(input) && typeof input[1] === 'string') {
                    usedBlockIds.add(input[1]);
                }
            }
        }

        const topLevelBlocks = Object.values(currentScript.blocks)
            .filter(b =>
                !usedBlockIds.has(b.id) &&
                !b.topLevel && !b.shadow &&
                b.id !== currentScript.topBlock &&
                !b.parent
            );

        connectBlocks(topLevelBlocks);

        // Attach first to event block
        const topEvent = currentScript.blocks[currentScript.topBlock];
        if (topLevelBlocks.length > 0) {
            topEvent.next = topLevelBlocks[0].id;
            topLevelBlocks[0].parent = topEvent.id;
        }

        scripts.push(currentScript);
    }
    if (!matched) {
        return [];
    }
    return scripts;
};

module.exports = parsePseudoCode;
