const path = require("path");
const webpack = require("webpack");

module.exports = {
  entry: "./src/index.js",
  output: {
    path: path.resolve(__dirname, '../frontend/static/frontend'),
    filename: "[name].js",
    publicPath: "/static/frontend/",
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
        },
      },
    ],
  },
  optimization: {
    minimize: true,
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env": {
        // This has effect on the react lib size
        'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'production')
      },
    }),
  ],
  devServer: {
    static: {
      directory: path.join(__dirname, '../frontend/static/frontend'),
    },
    devMiddleware: {
      writeToDisk: true, // This is the option that tells Webpack to write files to disk
    },
  },
};