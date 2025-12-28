// Package download - the downloader of hte model
package download

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type WriteCounter struct {
	Total int64 // Total bytes written
	Size  int64 // Total file size
	last  time.Time
}

// Write implements the io.Writer interface.
func (wc *WriteCounter) Write(p []byte) (n int, err error) {
	n = len(p)
	wc.Total += int64(n)
	now := time.Now()
	if now.Sub(wc.last) > 500*time.Millisecond {
		wc.PrintProgress()
		wc.last = now
	}
	return
}

// PrintProgress displays the download percentage
func (wc *WriteCounter) PrintProgress() {
	if wc.Size > 0 {
		percent := float64(wc.Total) / float64(wc.Size) * 100
		// \r returns the cursor to the start of the line, allowing overwrite
		fmt.Printf("\rDownloading... %.2f%% (%s / %s)",
			percent,
			byteCountToHuman(wc.Total),
			byteCountToHuman(wc.Size))
	} else {
		// Fallback if Content-Length is missing
		fmt.Printf("\rDownloading... %s (size unknown)", byteCountToHuman(wc.Total))
	}
}

func byteCountToHuman(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// DownloadFile handles the download, saving it to a local file.
func DownloadFile(url, destPath string) error {
	fmt.Printf("Attempting to download from: %s\n", url)

	// 1. HTTP GET request
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to make HTTP request %v", err)
	}
	defer func() {
		err := resp.Body.Close()
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad HTTP status: %v", resp.Status)
	}

	// 2. Setup the output file
	out, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", destPath, err)
	}

	defer func() {
		err := out.Close()
		if err != nil {
			fmt.Println(err.Error())
			os.Exit(1)

		}
	}()

	// 3. Write and track progress
	counter := &WriteCounter{Size: resp.ContentLength}

	// TeeReader pipes the data through the counter while copying to the file
	_, err = io.Copy(out, io.TeeReader(resp.Body, counter))
	if err != nil {
		return fmt.Errorf("failed to write file content %v", err)
	}

	// Final progress update to ensure 100% is displayed
	counter.Total = counter.Size // Force total to size for final display
	counter.PrintProgress()
	fmt.Println("\nDownload complete.")
	return nil
}
