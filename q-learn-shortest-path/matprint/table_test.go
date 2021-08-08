package matprint

import "testing"

func Test_digitCount(t *testing.T) {
	type args struct {
		n int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			name: "should return 1 for 0",
			args: args{
				n: 0,
			},
			want: 1,
		},
		{
			name: "should return 1 for 1",
			args: args{
				n: 1,
			},
			want: 1,
		},
		{
			name: "should return 1 for -1",
			args: args{
				n: -1,
			},
			want: 1,
		},
		{
			name: "should return 1 for 9",
			args: args{
				n: 9,
			},
			want: 1,
		},
		{
			name: "should return 1 for -9",
			args: args{
				n: -9,
			},
			want: 1,
		},
		{
			name: "should return 2 for 10",
			args: args{
				n: 10,
			},
			want: 2,
		},
		{
			name: "should return 2 for -10",
			args: args{
				n: -10,
			},
			want: 2,
		},
		{
			name: "should return 2 for 99",
			args: args{
				n: 99,
			},
			want: 2,
		},
		{
			name: "should return 10 for 1234567890",
			args: args{
				n: 1234567890,
			},
			want: 10,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := digitCount(tt.args.n); got != tt.want {
				t.Errorf("digitCount() = %v, want %v", got, tt.want)
			}
		})
	}
}
