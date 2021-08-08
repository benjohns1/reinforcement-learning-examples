package matprint

import (
	"fmt"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// TableCfg configures the string print function.
type TableCfg struct {
	Pad       int
	Separator string
	Precision int
	RowHeader bool
	ColHeader bool
}

// MaxPadding sets the amount of padding per cell.
func MaxPadding(pad int) func(cfg *TableCfg) {
	return func(cfg *TableCfg) {
		cfg.Pad = pad
	}
}

// Separator sets the cell separator string.
func Separator(sep string) func(cfg *TableCfg) {
	return func(cfg *TableCfg) {
		cfg.Separator = sep
	}
}

// Precision sets the display precision for the cell float value.
func Precision(p int) func(cfg *TableCfg) {
	return func(cfg *TableCfg) {
		cfg.Precision = p
	}
}

// Headers shows or hides column and row headers.
func Headers(row, col bool) func(cfg *TableCfg) {
	return func(cfg *TableCfg) {
		cfg.RowHeader = row
		cfg.ColHeader = col
	}
}

// NewTableCfg instantiates a new table printer configuration.
func NewTableCfg(opts ...func(cfg *TableCfg)) TableCfg {
	cfg := TableCfg{
		Separator: " ",
		Precision: 2,
		ColHeader: true,
		RowHeader: true,
	}
	applyOpts(&cfg, opts)
	return cfg
}

func applyOpts(cfg *TableCfg, opts []func(cfg *TableCfg)) {
	for _, opt := range opts {
		opt(cfg)
	}
}

func findMaxWidth(m mat.Matrix, cfg TableCfg, r, c, max int) int {
	formatter := fmt.Sprintf("%%.%df", cfg.Precision)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			l := len(fmt.Sprintf(formatter, m.At(i, j)))
			if l > max {
				max = l
			}
		}
	}
	return max
}

// Table prints a matrix in tabular form to a string with the given configuration.
func (t TableCfg) Table(m mat.Matrix, opts ...func(cfg *TableCfg)) string {
	cfg := t
	applyOpts(&cfg, opts)

	r, c := m.Dims()
	var headerColMaxWidth int
	if cfg.ColHeader {
		headerColMaxWidth = digitCount(c - 1)
	}
	if cfg.Pad == 0 {
		cfg.Pad = findMaxWidth(m, cfg, r, c, headerColMaxWidth)
	}

	formatter := fmt.Sprintf("%%%d.%df", cfg.Pad, cfg.Precision)
	rowStrs := make([]string, 0, r+2)
	for i := 0; i < r; i++ {
		cells := make([]string, 0, c)
		for j := 0; j < c; j++ {
			cells = append(cells, fmt.Sprintf(formatter, m.At(i, j)))
		}
		rowStrs = append(rowStrs, strings.Join(cells, cfg.Separator))
	}
	var hPad int
	if cfg.RowHeader {
		rFormat := fmt.Sprintf("%%%dd%s", headerColMaxWidth, " |")
		hPad = len(fmt.Sprintf(rFormat, 0))
		for i, str := range rowStrs {
			rowStrs[i] = fmt.Sprintf(rFormat, i) + str
		}
	}

	if cfg.ColHeader {
		rowStrs = append(header(c, hPad, cfg.Pad, cfg.Separator), rowStrs...)
	}
	return strings.Join(rowStrs, "\n") + "\n"
}

// Table prints a matrix in tabular form to a string.
func Table(m mat.Matrix, opts ...func(cfg *TableCfg)) string {
	return NewTableCfg().Table(m, opts...)
}

func digitCount(n int) int {
	if n == 0 {
		return 1
	}
	c := 0
	for n != 0 {
		n /= 10
		c += 1
	}
	return c
}

func header(cols, prepad int, pad int, sep string) []string {
	hFormatter := fmt.Sprintf("%%%ds", pad)
	hs := make([]string, 0, cols)
	var divLen int
	for j := 0; j < cols; j++ {
		h := fmt.Sprintf(hFormatter, strconv.Itoa(j))
		hs = append(hs, h)
		divLen += len(h)
	}
	var prepadStr string
	for i := prepad; i > 0; i-- {
		prepadStr += " "
	}
	numbers := fmt.Sprintf("%s%s", prepadStr, strings.Join(hs, sep))

	var divider string
	divLen += len(sep)*(cols-1) + prepad
	for i := divLen; i > 0; i-- {
		divider += "-"
	}
	return []string{numbers, divider}
}
